#!/usr/bin/env python3
"""In-silico region perturbation XAI for the multi-query methylation model.

For every region in a BED file:
  1. expand (optionally) to the model's 16-kb input window, centred on the region;
  2. perturb ONE modality *inside the original BED interval* to
       min  (0) | max (1 for ATAC/RNA, 100 for 5mC/5hmC) | random donor values;
  3. apply the learned [POS-MASK] token to the OTHER modalities at the CpG
     positions inside the BED interval, while keeping the perturbed modality
     visible;
  4. run the model and save the reconstructed values of the masked modalities
     over the whole 16-kb window, plus a per-region summary and the reference
     (unperturbed) prediction.

Important: the current model was trained with a *shared* position mask for all
tracks.  This script applies the position mask selectively at inference, so
"perturbed track visible + other tracks masked" is an out-of-distribution probe,
not a capability the model was trained for.  Results should be reported as a
counterfactual stress test.

The model is NOT modified.  A local forward re-implements the per-query stream
with per-track position masks (same submodules, same context fusion).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import (
    LazyM5cSequenceAtacRnaDataset,
    find_cpg_candidate_positions,
    find_forward_cpg_positions,
)
from models import MultiQueryMaskedPretrainingModel

CONTEXT_ORDER = ["5mc", "5hmc", "atac", "rna"]


# ---------------------------------------------------------------------------
# Dataset wrapper (same as run_multiquery_pretraining.MultiQueryDataset)
# ---------------------------------------------------------------------------
class MultiQueryDataset(torch.utils.data.Dataset):
    def __init__(self, base, track_order):
        self.base = base
        self.track_order = list(track_order)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        m5c, sequence_onehot, atac, hm5c, _loss_mask, rna = self.base[idx]
        source = {"5mc": m5c, "atac": atac, "5hmc": hm5c, "rna": rna}
        tracks = [source[name] for name in self.track_order]
        base_ids = sequence_onehot.argmax(dim=-1).long()
        return (*tracks, sequence_onehot, base_ids)


# ---------------------------------------------------------------------------
# Checkpoint / model
# ---------------------------------------------------------------------------
def resolve_tracks(args: dict) -> tuple[list[str], list[str]]:
    query = list(args["query_tracks"])
    available = {"5mc", "5hmc"}
    if args.get("atac_bw"):
        available.add("atac")
    if args.get("rna_coverage_bw"):
        available.add("rna")
    context = [t for t in CONTEXT_ORDER if t in available and t not in query]
    return query, context


def build_model(args: dict, device: torch.device):
    query, context = resolve_tracks(args)
    model = MultiQueryMaskedPretrainingModel(
        seq_len=int(args["target_length"]),
        hidden_dim=int(args["hidden_dim"]),
        use_positional_encoding=bool(args.get("use_positional_encoding", False)),
        num_blocks=int(args["num_blocks"]),
        fusion_type=args["fusion_type"],
        num_queries=len(query),
        num_context_tracks=len(context),
        context_mode=args.get("context_mode", "shared"),
    ).to(device)
    return model, query, context


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, query, context = build_model(ckpt["args"], device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return ckpt, model, query, context


# ---------------------------------------------------------------------------
# BED handling
# ---------------------------------------------------------------------------
def read_bed(path: str, expand_16k: bool, target_len: int, genome_fasta: str) -> pd.DataFrame:
    import pysam

    fa = pysam.FastaFile(genome_fasta)
    lengths = {name: int(length) for name, length in zip(fa.references, fa.lengths)}
    fa.close()

    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            if chrom not in lengths:
                continue
            start, end = sorted((int(parts[1]), int(parts[2])))
            name = parts[3] if len(parts) > 3 and parts[3] else f"{chrom}:{start}-{end}"
            center = (start + end) // 2

            if expand_16k:
                ws = center - target_len // 2
                we = ws + target_len
            else:
                ws = start
                we = start + target_len

            chrom_len = lengths[chrom]
            if ws < 0:
                ws, we = 0, target_len
            if we > chrom_len:
                we = chrom_len
                ws = max(0, we - target_len)
            if we - ws < target_len:
                continue

            rel_start = max(0, start - ws)
            rel_end = min(target_len, end - ws)
            if rel_end <= rel_start:
                continue

            rows.append({
                "chr": chrom,
                "start_expanded": int(ws),
                "end_expanded": int(we),
                "start": int(start),
                "end": int(end),
                "center": int(center),
                "name": str(name),
                "rel_start": int(rel_start),
                "rel_end": int(rel_end),
            })
    if not rows:
        raise ValueError("no usable regions in BED")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Selective-mask forward (same math as MultiQueryMaskedPretrainingModel)
# ---------------------------------------------------------------------------
def forward_with_selective_posmask(
    model,
    query_tracks: list[str],
    context_tracks: list[str],
    track_tensors: list[torch.Tensor],
    sequence: torch.Tensor,
    pos_masks: dict[str, Optional[torch.Tensor]],
):
    """track_tensors: [query..., context...] each (N,L,1); pos_masks: name -> mask or None."""
    nq, nc = len(query_tracks), len(context_tracks)
    track_hidden = []
    for i, name in enumerate(query_tracks):
        h = model.query_norms[i](model.query_projs[i](track_tensors[i]))
        mask = pos_masks.get(name)
        if mask is not None:
            h = h * (1.0 - mask) + model.query_pos_tokens[i].view(1, 1, -1) * mask
        track_hidden.append(h)
    for k, name in enumerate(context_tracks):
        h = model.context_track_norms[k](model.context_track_projs[k](track_tensors[nq + k]))
        mask = pos_masks.get(name)
        if mask is not None:
            h = h * (1.0 - mask) + model.context_pos_tokens[k].view(1, 1, -1) * mask
        track_hidden.append(h)

    sequence_hidden = model.sequence_norm(model.sequence_proj(sequence))

    context_cache = {}
    preds = []
    for i in range(nq):
        key = tuple(model.context_specs[i])
        context = context_cache.get(key)
        if context is None:
            context = model._fuse_context(sequence_hidden, track_hidden, model.context_specs[i])
            context_cache[key] = context
        hidden = track_hidden[i]
        if model.position_encoding is not None:
            hidden = model.position_encoding(hidden)
        for block in model.blocks:
            hidden = block(hidden, context)
        hidden = model.final_norms[i](hidden)
        preds.append(model.heads[i](hidden))
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="In-silico region perturbation XAI")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bed", required=True)
    ap.add_argument("--perturb-track", required=True, choices=CONTEXT_ORDER)
    ap.add_argument("--mode", required=True, choices=["min", "max", "random"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--expand-16k", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mask-others", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-value", type=float, default=None,
                    help="value for mode=max; default 1.0 for atac/rna, 100.0 for 5mc/5hmc")
    ap.add_argument("--max-regions", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--genome-fasta", default=None)
    ap.add_argument("--m5c-bedgraph", default=None)
    ap.add_argument("--hm5c-bedgraph", default=None)
    ap.add_argument("--atac-bw", default=None)
    ap.add_argument("--rna-bw", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    ckpt, model, query_tracks, context_tracks = load_checkpoint(args.checkpoint, device)
    ck_args = ckpt["args"]
    track_order = query_tracks + context_tracks
    if args.perturb_track not in track_order:
        raise ValueError(f"perturb-track {args.perturb_track} not in model tracks {track_order}")

    genome_fasta = args.genome_fasta or ck_args["genome_fasta"]
    m5c_path = args.m5c_bedgraph or ck_args["m5c_bedgraph"][0]
    hm5c_path = args.hm5c_bedgraph or ck_args["hm5c_bedgraph"][0]
    atac_path = args.atac_bw or ck_args["atac_bw"][0]
    rna_path = args.rna_bw or ck_args["rna_coverage_bw"][0]

    df = read_bed(args.bed, args.expand_16k, int(ck_args["target_length"]), genome_fasta)
    if args.max_regions and len(df) > args.max_regions:
        df = df.iloc[: args.max_regions].reset_index(drop=True)
    print(f"[bed] {len(df)} regions | expand16k={args.expand_16k} | mask_others={args.mask_others}")

    base = LazyM5cSequenceAtacRnaDataset(
        indices=list(range(len(df))),
        df_dmr=df,
        genome_fasta=genome_fasta,
        m5c_bedgraph=m5c_path,
        hm5c_bedgraph=hm5c_path,
        atac_bw_path=atac_path,
        target_length=int(ck_args["target_length"]),
        mask_mode=ck_args.get("mask_mode", "cpg_forward"),
        atac_scaling=ck_args.get("atac_scaling", "minmax"),
        rna_bw_path=rna_path,
        rna_scaling=ck_args.get("rna_scaling", "minmax"),
        augment_rc=False,
        clip_at_zero=bool(ck_args.get("clip_at_zero", False)),
    )
    ds = MultiQueryDataset(base, track_order)

    # Load all regions (bounded by --max-regions).
    all_tracks = [[] for _ in track_order]
    all_seq, all_base = [], []
    for i in range(len(ds)):
        item = ds[i]
        for j in range(len(track_order)):
            all_tracks[j].append(item[j])
        all_seq.append(item[len(track_order)])
        all_base.append(item[len(track_order) + 1])
    all_tracks = [torch.stack(t, 0) for t in all_tracks]  # (N,L,1)
    all_seq = torch.stack(all_seq, 0)                     # (N,L,4)
    all_base = torch.stack(all_base, 0)                   # (N,L)
    N, L = all_base.shape
    print(f"[data] N={N} L={L}")
    if args.mode == "random" and N < 2:
        raise ValueError("--mode random needs at least 2 regions (for a donor window)")

    perturb_idx = track_order.index(args.perturb_track)
    max_value = args.max_value
    if max_value is None:
        max_value = 1.0 if args.perturb_track in ("atac", "rna") else 100.0

    # Per-region relative interval and CpG mask inside the BED interval.
    rel_start = df["rel_start"].to_numpy()
    rel_end = df["rel_end"].to_numpy()
    cpg = find_cpg_candidate_positions(all_base)            # (N,L) bool
    fwd_c = find_forward_cpg_positions(all_base)            # (N,L) bool

    # Selected (masked) tracks: every track except the perturbed one.
    masked_tracks = [t for t in query_tracks if t != args.perturb_track]
    masked_context = [t for t in context_tracks if t != args.perturb_track and args.mask_others]
    if masked_context:
        print(f"[warn] masked context tracks {masked_context} have no output head; only query heads are saved")

    # Output arrays (only query heads can be reconstructed).
    out_pred = {t: np.full((N, L), np.nan, dtype=np.float32) for t in masked_tracks}
    out_ref = {t: np.full((N, L), np.nan, dtype=np.float32) for t in masked_tracks}
    summary = []

    for b0 in range(0, N, args.batch_size):
        b1 = min(N, b0 + args.batch_size)
        idx = list(range(b0, b1))
        nb = len(idx)

        def make_batch(tracks_override=None):
            tracks = []
            for j in range(len(track_order)):
                t = all_tracks[j][idx].clone()
                tracks.append(t.to(device))
            if tracks_override is not None:
                tracks[perturb_idx] = tracks_override
            seq = all_seq[idx].to(device)
            return tracks, seq

        # Selective position masks: CpGs inside the BED interval, for all tracks
        # except the perturbed one.  The perturbed track stays visible.
        pos_masks = {}
        for t in track_order:
            if t == args.perturb_track or not args.mask_others:
                pos_masks[t] = None
                continue
            m = torch.zeros(nb, L, 1, dtype=torch.float32)
            for k, i in enumerate(idx):
                rs, re = int(rel_start[i]), int(rel_end[i])
                if re > rs:
                    m[k, rs:re, 0] = cpg[i, rs:re].float()
            pos_masks[t] = m.to(device)

        # Reference: original perturbed-track values, SAME masks as the perturbed run.
        ref_tracks, seq = make_batch()
        ref_preds = forward_with_selective_posmask(
            model, query_tracks, context_tracks, ref_tracks, seq, pos_masks,
        )

        # Perturbed batch: copy, then overwrite the perturbed track in the BED interval.
        pert_track = all_tracks[perturb_idx][idx].clone()
        for k, i in enumerate(idx):
            rs, re = int(rel_start[i]), int(rel_end[i])
            if re <= rs:
                continue
            if args.mode == "min":
                pert_track[k, rs:re, 0] = 0.0
            elif args.mode == "max":
                pert_track[k, rs:re, 0] = max_value
            else:
                donor = int(np.random.randint(0, N - 1))
                if donor >= i:
                    donor += 1
                need = re - rs
                off = int(np.random.randint(0, L - need + 1))
                pert_track[k, rs:re, 0] = all_tracks[perturb_idx][donor, off:off + need, 0]
        pert_tracks, _ = make_batch(pert_track.to(device))

        pert_preds = forward_with_selective_posmask(
            model, query_tracks, context_tracks, pert_tracks, seq, pos_masks,
        )

        for t in masked_tracks:
            qi = query_tracks.index(t)
            pr = pert_preds[qi][..., 0].detach().cpu().numpy()
            rr = ref_preds[qi][..., 0].detach().cpu().numpy()
            for k, i in enumerate(idx):
                out_pred[t][i] = pr[k]
                out_ref[t][i] = rr[k]
                rs, re = int(rel_start[i]), int(rel_end[i])
                sel = fwd_c[i, rs:re].numpy() if re > rs else np.zeros(0, dtype=bool)
                if sel.any():
                    pv = pr[k, rs:re][sel]
                    rv = rr[k, rs:re][sel]
                    summary.append({
                        "name": df.iloc[i]["name"],
                        "chr": df.iloc[i]["chr"],
                        "start": int(df.iloc[i]["start"]),
                        "end": int(df.iloc[i]["end"]),
                        "perturb_track": args.perturb_track,
                        "mode": args.mode,
                        "masked_track": t,
                        "n_cpg": int(sel.sum()),
                        "mean_ref": float(rv.mean()),
                        "mean_pert": float(pv.mean()),
                        "delta": float(pv.mean() - rv.mean()),
                    })
                else:
                    summary.append({
                        "name": df.iloc[i]["name"],
                        "chr": df.iloc[i]["chr"],
                        "start": int(df.iloc[i]["start"]),
                        "end": int(df.iloc[i]["end"]),
                        "perturb_track": args.perturb_track,
                        "mode": args.mode,
                        "masked_track": t,
                        "n_cpg": 0,
                        "mean_ref": np.nan,
                        "mean_pert": np.nan,
                        "delta": np.nan,
                    })

    # ---- save ----
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.out_dir, f"{ts}_region_perturbation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    npz_path = os.path.join(args.out_dir, f"{ts}_region_perturbation_predictions.npz")
    payload = {
        "chrom": df["chr"].to_numpy(),
        "start": df["start"].to_numpy(),
        "end": df["end"].to_numpy(),
        "name": df["name"].to_numpy(),
        "window_start": df["start_expanded"].to_numpy(),
        "window_end": df["end_expanded"].to_numpy(),
        "rel_start": rel_start,
        "rel_end": rel_end,
        "base_ids": all_base.numpy().astype(np.int8),
        "perturb_track": np.array(args.perturb_track),
        "mode": np.array(args.mode),
        "max_value": np.array(max_value),
    }
    for t in masked_tracks:
        payload[f"pred_{t}"] = out_pred[t]
        payload[f"ref_{t}"] = out_ref[t]
    np.savez_compressed(npz_path, **payload)

    manifest = {
        "timestamp": ts,
        "checkpoint": args.checkpoint,
        "bed": args.bed,
        "out_dir": args.out_dir,
        "device": str(device),
        "query_tracks": query_tracks,
        "context_tracks": context_tracks,
        "perturb_track": args.perturb_track,
        "mode": args.mode,
        "max_value": max_value,
        "expand_16k": args.expand_16k,
        "mask_others": args.mask_others,
        "n_regions": int(N),
        "seq_len": int(L),
        "n_summary_rows": int(len(summary_df)),
        "summary_csv": summary_path,
        "predictions_npz": npz_path,
        "note": "Selective position mask at inference. The model was trained with a shared mask for all tracks, so 'perturbed track visible + others masked' is an out-of-distribution probe.",
    }
    manifest_path = os.path.join(args.out_dir, f"{ts}_region_perturbation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"[done] {summary_path}")
    print(f"[done] {npz_path}")
    print(f"[done] {manifest_path}")


if __name__ == "__main__":
    main()
