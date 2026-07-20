#!/usr/bin/env python
"""
Ablation training script: explore which track plays the role of "query" vs "context"
for 5hmC prediction. Uses FlexibleQueryRegressorModelB.

Usage:
  python run_atac_ablation.py --ablation atac_query_only [other args]

Ablation presets (use --ablation):
  m5c_atac       : baseline    — query=m5c, context=[atac]                  (cross-hyena fusion by default)
  m5c_atac_attn  : attn-only   — query=m5c, context=[atac]                  (forces --model-b-fusion=cross_attention)
  atac_m5c       : swapped     — query=atac, context=[m5c]
  atac_only      : ATAC-only   — query=atac, context=[]                      (no 5mC input)
  m5c_only       : 5mC-only    — query=m5c, context=[]                      (no ATAC input)
  seq_query      : seq-as-q    — query=sequence, context=[atac, m5c]
  m5c_atac_q     : concat-q    — query=concat(m5c,atac), context=[]
  seq_only       : seq-only    — query=sequence, context=[]                  (text-only baseline)
  atac_seq       : alt         — query=atac, context=[sequence]
  all_three      : all         — query=concat(m5c,atac), context=[]          (DNA via dedicated track; was context=[sequence], but ModelB hardcodes context_track_dim=1)

Note: ablations whose name ends in ``_attn`` force ``cross_attention`` fusion
regardless of ``--model-b-fusion``. Add new ones by appending to
``ABLATIONS`` and (optionally) ``ABLATION_FUSION_OVERRIDES``.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import pyfaidx

from data import (
    LazyM5cSequenceAtacDataset,
    assign_non_overlapping_groups,
    ensure_path_list,
    get_sequence,
    load_data,
    resolve_loss_mask,
    write_train_val_beds,
)
from models import FlexibleQueryRegressorModelB
from utils import (
    export_prediction_signals,
    get_freest_gpu,
    plot_regression_predictions,
    resolve_sample_sizes,
    set_random_seed,
)


# ---------------------------------------------------------------------------
# Ablation configuration presets
# ---------------------------------------------------------------------------

ABLATIONS = {
    # name            : (query_modality, list-of-context-modalities)
    "m5c_atac":      ("m5c",    ["atac"]),
    "m5c_atac_attn": ("m5c",    ["atac"]),       # same as m5c_atac but forces cross_attention fusion
    "atac_m5c":      ("atac",   ["m5c"]),
    "atac_only":     ("atac",   []),
    "m5c_only":      ("m5c",    []),
    "seq_query":     ("sequence", ["atac", "m5c"]),
    "m5c_atac_q":    ("m5c_atac", []),                # concat(m5c, atac) as query
    "seq_only":      ("sequence", []),
    "atac_seq":      ("atac",   ["sequence"]),
    "all_three":     ("m5c_atac", []),                # concat(m5c,atac) as query, sequence via dedicated track
}


# Ablations whose name ends in "_attn" force cross_attention fusion,
# overriding --model-b-fusion.  Keys here take precedence over the convention.
ABLATION_FUSION_OVERRIDES = {
    "m5c_atac_attn": "cross_attention",
}


def get_ablation_fusion(ablation_name: str, default_fusion: str) -> str:
    """Return the fusion type for ``ablation_name``.

    Priority:
      1. explicit entry in ``ABLATION_FUSION_OVERRIDES``
      2. ``_attn`` suffix → cross_attention
      3. ``default_fusion`` (from --model-b-fusion)
    """
    if ablation_name in ABLATION_FUSION_OVERRIDES:
        return ABLATION_FUSION_OVERRIDES[ablation_name]
    if ablation_name.endswith("_attn"):
        return "cross_attention"
    return default_fusion


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def masked_mse_loss(pred, target, mask):
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def build_context_list(args, ablation_name):
    """Return list of modality names used as context (excluding sequence)."""
    return ABLATIONS[ablation_name][1]


def get_query_dim(ablation_name: str) -> int:
    """Return dimensionality of the query input."""
    q = ABLATIONS[ablation_name][0]
    if q == "m5c" or q == "atac":
        return 1
    if q == "sequence":
        return 4
    if q == "m5c_atac":
        return 2
    raise ValueError(f"Unknown query modality: {q}")


# ---------------------------------------------------------------------------
# Flexible Dataset wrapper
# ---------------------------------------------------------------------------

class FlexibleAblationDataset(torch.utils.data.Dataset):
    """Wraps LazyM5cSequenceAtacDataset and selects / reshapes tensors based on
    the chosen ablation (which modality is query, which are context)."""

    def __init__(self, base_ds: LazyM5cSequenceAtacDataset, ablation_name: str):
        self.base_ds = base_ds
        self.ablation_name = ablation_name
        # Cache the original 5-tuple: (m5c, seq_onehot, atac, hm5c_target, mask)
        # Indices into base_ds[i]:
        #   0=m5c, 1=sequence_onehot, 2=atac, 3=hm5c_target, 4=loss_mask
        q, ctx = ABLATIONS[ablation_name]
        self._q_kind = q
        self._ctx_kinds = ctx

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        m5c, seq, atac, target, mask = self.base_ds[idx]
        # m5c, atac: (L, 1) float
        # seq:        (L, 4) float one-hot
        # target:     (L, 1) float (5hmC)
        # mask:       (L, 1) float

        # ---- query ----
        if self._q_kind == "m5c":
            query = m5c
        elif self._q_kind == "atac":
            query = atac
        elif self._q_kind == "sequence":
            query = seq
        elif self._q_kind == "m5c_atac":
            query = torch.cat([m5c, atac], dim=-1)
        else:
            raise ValueError(f"Unknown query: {self._q_kind}")

        # ---- context tracks (excluding sequence) ----
        ctx_tracks = []
        for kind in self._ctx_kinds:
            if kind == "m5c":
                ctx_tracks.append(m5c)
            elif kind == "atac":
                ctx_tracks.append(atac)
            else:
                raise ValueError(f"Unknown context modality: {kind}")

        return query, seq, ctx_tracks, target, mask


def flexible_collate(batch):
    """Collate function: pads/stacks context lists into a single tensor."""
    queries, seqs, ctx_lists, targets, masks = zip(*batch)
    queries = torch.stack(queries, dim=0)
    seqs = torch.stack(seqs, dim=0)
    targets = torch.stack(targets, dim=0)
    masks = torch.stack(masks, dim=0)
    # Stack context tracks across batch
    num_ctx = len(ctx_lists[0])
    ctx_batched = []
    for i in range(num_ctx):
        ctx_batched.append(torch.stack([cl[i] for cl in ctx_lists], dim=0))
    return queries, seqs, ctx_batched, targets, masks


# ---------------------------------------------------------------------------
# Eval / predict
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    total_loss = total_count = 0
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for q, s, c, t, m in loader:
            q = q.to(device)
            s = s.to(device)
            t = t.to(device)
            m = m.to(device)
            c = [ct.to(device) for ct in c]
            p = model(q, s, c)
            loss = masked_mse_loss(p, t, m)
            total_loss += loss.item() * q.size(0)
            total_count += q.size(0)
            preds.append(p.cpu())
            targets.append(t.cpu())
            masks.append(m.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    masks = torch.cat(masks)
    ss_res = (((targets - preds) ** 2) * masks).sum()
    mt = targets[masks.bool()]
    mp = preds[masks.bool()]
    mean = mt.mean() if mt.numel() > 0 else torch.tensor(0.0)
    ss_tot = (((targets - mean) ** 2) * masks).sum().clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    if mt.numel() > 1:
        ct = mt - mt.mean()
        cp = mp - mp.mean()
        denom = ct.pow(2).sum().sqrt() * cp.pow(2).sum().sqrt()
        pearson = (ct * cp).sum() / denom.clamp_min(1e-12)
    else:
        pearson = float("nan")
    return total_loss / max(total_count, 1), r2.item(), pearson.item()


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ablation models.")
    p.add_argument("--ablation", choices=list(ABLATIONS.keys()), default="atac_only",
                   help="Which ablation preset to run.")

    # data
    p.add_argument("--dmr-csv", default="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed")
    p.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    p.add_argument("--m5c-bedgraph", nargs="+", required=True)
    p.add_argument("--hm5c-bedgraph", nargs="+", required=True)
    p.add_argument("--atac-bw", nargs="+", required=True)
    p.add_argument("--chromosome", default=None)
    p.add_argument("--target-length", type=int, default=16384)
    p.add_argument("--sample-sizes", nargs="+", default=["all"])
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mask-mode", choices=["cpg_both", "cpg_forward", "all"], default="cpg_forward")
    p.add_argument("--atac-scaling", choices=["none", "minmax"], default="minmax")

    # model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--model-b-blocks", type=int, default=2)
    p.add_argument("--model-b-fusion", choices=["cross_hyena", "cross_attention"], default="cross_hyena")
    p.add_argument("--use-positional-encoding", action="store_true")
    p.add_argument("--augment-reverse-complement", action="store_true")

    # training
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    p.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-patience", type=int, default=2)
    p.add_argument("--scheduler-t-max", type=int, default=0)

    # optimization
    p.add_argument("--amp", action="store_true")
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lazy", action="store_true")

    # output
    p.add_argument("--output-dir", default="output/ablation")
    p.add_argument("--timestamp", default="")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.timestamp:
        from datetime import datetime
        args.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    ablation_name = args.ablation
    q_mod, ctx_mods = ABLATIONS[ablation_name]
    print(f"============================================")
    print(f"[Ablation] {ablation_name}")
    print(f"  query     : {q_mod}")
    print(f"  context   : {ctx_mods} (+sequence always)")
    print(f"============================================")

    # Validate required data files
    if q_mod in ("m5c", "m5c_atac") or "m5c" in ctx_mods:
        if not args.m5c_bedgraph:
            raise ValueError(f"Ablation {ablation_name} needs --m5c-bedgraph")
    if q_mod in ("atac", "m5c_atac") or "atac" in ctx_mods:
        if not args.atac_bw:
            raise ValueError(f"Ablation {ablation_name} needs --atac-bw")

    # Resolve sample sizes
    sample_sizes = resolve_sample_sizes(args.sample_sizes, args)
    args.sample_sizes = sample_sizes  # update for downstream data loaders
    usable_dmrs = sample_sizes[0]
    print(f"  samples   : {usable_dmrs if usable_dmrs != float('inf') else 'all'}")

    set_random_seed(args.seed)
    args.use_m5c = True
    if args.chromosome is not None:
        args.chromosome = args.chromosome  # normalize if needed

    # Load data (lazy mode if --lazy)
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(
        args, lazy=args.lazy,
    )
    if len(df_dmr) > usable_dmrs:
        df_dmr = df_dmr.iloc[:usable_dmrs].copy().reset_index(drop=True)

    seq_len = args.target_length

    # Build base Dataset (LazyM5cSequenceAtacDataset) + flexible wrapper
    if args.lazy:
        # Build per-split indices using the non-overlap-group split
        split_regions_df = df_dmr.copy().reset_index().rename(columns={"index": "original_idx"})
        split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
        split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
        split_regions_df = assign_non_overlapping_groups(
            split_regions_df, "chr", "start_expanded", "end_expanded"
        )
        group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
        num_train_groups = max(1, int(len(group_ids) * args.train_ratio))
        train_group_ids = set(group_ids[:num_train_groups].tolist())
        train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
        train_indices = np.flatnonzero(train_mask).tolist()
        val_indices = np.flatnonzero(~train_mask).tolist()

        # Write train / val region BED files to output dir for inspection
        bed_out_dir = Path(args.output_dir) / ablation_name
        write_train_val_beds(
            split_regions_df,
            train_indices,
            val_indices,
            output_dir=bed_out_dir,
            timestamp=args.timestamp,
        )

        hm5c_paths = ensure_path_list(args.hm5c_bedgraph)
        m5c_paths = ensure_path_list(args.m5c_bedgraph)
        atac_paths = ensure_path_list(args.atac_bw)

        train_base = LazyM5cSequenceAtacDataset(
            indices=train_indices,
            df_dmr=split_regions_df,
            genome_fasta=args.genome_fasta,
            m5c_bedgraph=m5c_paths[0],
            hm5c_bedgraph=hm5c_paths[0],
            atac_bw_path=atac_paths[0],
            target_length=args.target_length,
            mask_mode=args.mask_mode,
            atac_scaling=args.atac_scaling,
            augment_rc=args.augment_reverse_complement,
        )
        val_base = LazyM5cSequenceAtacDataset(
            indices=val_indices,
            df_dmr=split_regions_df,
            genome_fasta=args.genome_fasta,
            m5c_bedgraph=m5c_paths[0],
            hm5c_bedgraph=hm5c_paths[0],
            atac_bw_path=atac_paths[0],
            target_length=args.target_length,
            mask_mode=args.mask_mode,
            atac_scaling=args.atac_scaling,
            augment_rc=False,
        )
    else:
        # Non-lazy: just use the whole loaded tensor set (no proper split, for fast dev tests)
        usable = min(usable_dmrs, len(df_dmr), len(seqs), len(mcg_tracks), len(hmcg_tracks), len(atac_tracks))
        train_base = val_base = None  # placeholder, will fall through to error
        raise NotImplementedError("Only --lazy mode is supported in this ablation script.")

    train_ds = FlexibleAblationDataset(train_base, ablation_name)
    val_ds = FlexibleAblationDataset(val_base, ablation_name)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=flexible_collate,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=flexible_collate,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True,
    )

    # Model
    device = torch.device(f"cuda:{get_freest_gpu()}" if torch.cuda.is_available() else "cpu")
    num_ctx_tracks = len(ctx_mods)
    fusion_type = get_ablation_fusion(ablation_name, args.model_b_fusion)
    if fusion_type != args.model_b_fusion:
        print(f"  [fusion override] {args.model_b_fusion} -> {fusion_type} (from ablation '{ablation_name}')")
    args.model_b_fusion = fusion_type  # persist resolved value into checkpoint + metrics
    model = FlexibleQueryRegressorModelB(
        seq_len=seq_len,
        query_dim=get_query_dim(ablation_name),
        sequence_dim=4,
        context_track_dim=1,  # each track has dim=1
        num_context_tracks=num_ctx_tracks,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=args.use_positional_encoding,
        num_blocks=args.model_b_blocks,
        fusion_type=fusion_type,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: FlexibleQueryRegressorModelB  |  query={q_mod}  ctx={ctx_mods}  |  fusion={fusion_type}  |  params: {n_params:,}")

    # Optimizer / scheduler / AMP
    decay_params = [p for n, p in model.named_parameters() if p.ndim > 1 and "norm" not in n.lower()]
    no_decay_params = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "norm" not in n.lower())]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.num_epochs), eta_min=args.scheduler_min_lr
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.scheduler_min_lr,
        )
    else:
        scheduler = None

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    amp_dtype = torch.bfloat16 if args.amp else torch.float32

    # Output dir
    out_dir = Path(args.output_dir) / ablation_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / f"{args.timestamp}_best.pt"
    last_ckpt = out_dir / f"{args.timestamp}_last.pt"

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = seen = 0
        optimizer.zero_grad()
        accum = 0
        for q, s, c, t, m in train_loader:
            q = q.to(device)
            s = s.to(device)
            t = t.to(device)
            m = m.to(device)
            c = [ct.to(device) for ct in c]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                p = model(q, s, c)
                loss = masked_mse_loss(p, t, m) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accum += 1
            running_loss += loss.item() * q.size(0) * args.gradient_accumulation_steps
            seen += q.size(0)
            if accum % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        train_loss = running_loss / max(seen, 1)
        val_loss, val_r2, val_pearson = evaluate(model, val_loader, device)
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{ablation_name}] epoch {epoch:02d} | train={train_loss:.4f} "
            f"| val={val_loss:.4f} | r2={val_r2:.4f} | pearson={val_pearson:.4f} | lr={cur_lr:.2e}"
        )
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state_dict": best_state, "ablation": ablation_name,
                        "args": vars(args)}, best_ckpt)
            patience_left = args.patience
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                print(f"Early stop at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"epoch": best_epoch, "model_state_dict": model.state_dict(),
                "ablation": ablation_name, "args": vars(args)}, last_ckpt)

    # Final eval + save metrics
    final_val_loss, final_val_r2, final_val_pearson = evaluate(model, val_loader, device)
    metrics = {
        "ablation": ablation_name,
        "query_modality": q_mod,
        "context_modalities": ctx_mods,
        "fusion_type": fusion_type,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_r2": final_val_r2,    # last-loaded best state metrics
        "best_val_pearson": final_val_pearson,
        "num_params": n_params,
        "num_train_regions": len(train_ds),
        "num_val_regions": len(val_ds),
    }
    metrics_path = out_dir / f"{args.timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()