#!/usr/bin/env python3
"""Generalized zero-shot evaluation of a 2-head 5mC/5hmC model on external data.

Masks a fraction of covered CpG C sites, optionally widens each mask to a
+/-bp hole (all bases), and saves per-CpG predictions for paired bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pyfaidx
import torch
from scipy.stats import spearmanr

from models import MultiQueryMaskedPretrainingModel

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def tabix_fetch(path, chrom, start, end):
    proc = subprocess.run(["tabix", path, f"{chrom}:{start + 1}-{end}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        f = line.split("\t")
        if len(f) < 4:
            continue
        try:
            s, e, v = int(f[1]), int(f[2]), float(f[3])
        except ValueError:
            continue
        if e <= start or s >= end:
            continue
        rows.append((max(s, start), min(e, end), v))
    return rows


def load_track_window(path, chrom, start, length, scale=1.0):
    arr = np.zeros(length, dtype=np.float32)
    covered = np.zeros(length, dtype=bool)
    for s, e, v in tabix_fetch(path, chrom, start, start + length):
        a, b = max(0, s - start), min(length, e - start)
        if b > a:
            arr[a:b] = float(v) * scale
            covered[a:b] = True
    return arr, covered


def one_hot_sequence(seq):
    base = np.full(len(seq), 4, dtype=np.int64)
    arr = np.array(list(seq))
    for b, i in BASE_MAP.items():
        base[arr == b] = i
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i in range(4):
        x[base == i, i] = 1.0
    return x, base


def r2_score(pred, target):
    if pred.size < 2:
        return float("nan")
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pearson(pred, target):
    if pred.size < 2:
        return float("nan")
    x, y = pred - pred.mean(), target - target.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / max(denom, 1e-12))


def load_regions_bed(path, chroms):
    allowed = set(chroms)
    rows = []
    with open(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#") or line.startswith("track"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 3:
                continue
            chrom = f[0]
            if chrom in allowed:
                gchrom = chrom
            elif chrom.startswith("chr") and chrom[3:] in allowed:
                gchrom = chrom[3:]
            elif ("chr" + chrom) in allowed:
                gchrom = "chr" + chrom
            else:
                continue
            data_chrom = chrom if chrom.startswith("chr") else "chr" + chrom
            try:
                s, e = int(float(f[1])), int(float(f[2]))
            except ValueError:
                continue
            if e > s:
                rows.append((gchrom, data_chrom, s, e))
    return rows


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    q = a.get("query_tracks") or ["5mc", "5hmc"]
    if list(q) != ["5mc", "5hmc"]:
        raise ValueError(f"expected query_tracks ['5mc','5hmc'], got {q}")
    model = MultiQueryMaskedPretrainingModel(
        seq_len=int(a.get("target_length", 16384)),
        hidden_dim=int(a.get("hidden_dim", 64)),
        use_positional_encoding=bool(a.get("use_positional_encoding", False)),
        num_blocks=int(a.get("num_blocks", 2)),
        fusion_type=a.get("fusion_type", "cross_hyena"),
        num_queries=2,
        num_context_tracks=2,
        context_mode=a.get("context_mode", "all-others"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, a, int(a.get("target_length", 16384))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--m5c", required=True)
    ap.add_argument("--hm5c", required=True)
    ap.add_argument("--genome", default="/data1st2/hannan_25/genome_ref/mm10/Mus_musculus.GRCm38.dna.primary_assembly.fa")
    ap.add_argument("--regions-bed", default="/data1st1/junyi/methdata/GSE279860_finetune_input/regions/cpg_test.bed")
    ap.add_argument("--mask-bp", type=int, default=50)
    ap.add_argument("--mask-fraction", type=float, default=0.15)
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--min-cg", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="zs")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, ckpt_args, seq_len = load_model(args.checkpoint, device)
    genome = pyfaidx.Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    chroms = [c for c in genome.keys()
              if c not in ("chrM", "chrMT", "M", "MT") and "random" not in c.lower()
              and len(genome[c]) > seq_len + 1000]
    regions = load_regions_bed(args.regions_bed, chroms)
    print(f"[zs] {args.tag} | windows={len(regions)} | mask_bp={args.mask_bp} | scale={args.value_scale}", flush=True)

    order = rng.permutation(len(regions))
    win_ids = []
    out = {k: [] for k in ("pos", "spacing", "t5mc", "p5mc", "t5hmc", "p5hmc")}
    win_meta = []
    kept = 0
    t0 = time.time()

    for wi in order:
        if kept >= args.max_windows:
            break
        gchrom, dchrom, rs, re = regions[wi]
        center = (rs + re) // 2
        start = max(0, center - seq_len // 2)
        if start + seq_len > len(genome[gchrom]):
            start = max(0, len(genome[gchrom]) - seq_len)
        seq = str(genome[gchrom][start:start + seq_len])
        if len(seq) != seq_len:
            continue
        seq_x, base = one_hot_sequence(seq)
        if float((base == 4).mean()) > 0.2:
            continue

        m5c_vals, cov_m = load_track_window(args.m5c, dchrom, start, seq_len, args.value_scale)
        h5c_vals, cov_h = load_track_window(args.hm5c, dchrom, start, seq_len, args.value_scale)
        cpg = (base == 1) & (np.roll(base, -1) == 2)
        cpg[-1] = False
        covered = cov_m & cov_h & cpg
        cov_idx = np.flatnonzero(covered)
        if cov_idx.size < args.min_cg:
            continue

        n_mask = max(1, int(round(args.mask_fraction * cov_idx.size)))
        seed_idx = rng.choice(cov_idx, size=n_mask, replace=False)
        seed_idx.sort()

        pos_mask = np.zeros((1, seq_len, 1), dtype=np.float32)
        if args.mask_bp <= 0:
            pos_mask[0, seed_idx, 0] = 1.0
        else:
            bp = args.mask_bp
            for k in seed_idx:
                pos_mask[0, max(0, k - bp):min(seq_len, k + bp + 1), 0] = 1.0

        # nearest other covered CpG distance (genomic spacing)
        spacing = np.full(seed_idx.size, 10**6, dtype=np.int32)
        for j, k in enumerate(seed_idx):
            d = np.abs(cov_idx - k)
            d[d == 0] = 10**9
            spacing[j] = int(d.min()) if d.size > 1 else 10**6

        q5mc = torch.from_numpy(m5c_vals).view(1, seq_len, 1).to(device)
        q5hmc = torch.from_numpy(h5c_vals).view(1, seq_len, 1).to(device)
        ctx = [torch.zeros_like(q5mc), torch.zeros_like(q5mc)]
        seq_t = torch.from_numpy(seq_x).unsqueeze(0).to(device)
        pos_t = torch.from_numpy(pos_mask).to(device)
        qp = [torch.ones(1, 1, 1, device=device)] * 2
        tp = [torch.zeros(1, 1, 1, device=device)] * 2

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            preds = model([q5mc, q5hmc], seq_t, ctx,
                          position_mask=pos_t, query_present=qp, track_present=tp)
        p5mc = preds[0][0, :, 0].float().cpu().numpy()[seed_idx]
        p5hmc = preds[1][0, :, 0].float().cpu().numpy()[seed_idx]

        out["pos"].append(seed_idx.astype(np.int32))
        out["spacing"].append(spacing)
        out["t5mc"].append(m5c_vals[seed_idx])
        out["p5mc"].append(p5mc.astype(np.float32))
        out["t5hmc"].append(h5c_vals[seed_idx])
        out["p5hmc"].append(p5hmc.astype(np.float32))
        win_ids.append(kept)
        win_meta.append((gchrom, int(start)))
        kept += 1
        if kept % 200 == 0:
            print(f"[zs] {args.tag} windows={kept} cpgs={sum(len(x) for x in out['pos'])} "
                  f"{time.time()-t0:.0f}s", flush=True)

    if kept == 0:
        raise SystemExit("no windows kept")

    cat = {k: np.concatenate(v) for k, v in out.items()}
    win = np.concatenate([np.full(len(p), w, dtype=np.int32) for w, p in zip(win_ids, out["pos"])])
    np.savez_compressed(args.out_npz, win=win, win_meta=np.array(win_meta, dtype=object), **cat)

    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "mask_bp": args.mask_bp,
        "mask_fraction": args.mask_fraction,
        "value_scale": args.value_scale,
        "n_windows": int(kept),
        "n_cpg": int(cat["pos"].size),
        "r2_5mc": r2_score(cat["p5mc"], cat["t5mc"]),
        "r2_5hmc": r2_score(cat["p5hmc"], cat["t5hmc"]),
        "rho_5mc": pearson(cat["p5mc"], cat["t5mc"]),
        "rho_5hmc": pearson(cat["p5hmc"], cat["t5hmc"]),
        "spearman_5mc": float(spearmanr(cat["p5mc"], cat["t5mc"]).correlation),
        "spearman_5hmc": float(spearmanr(cat["p5hmc"], cat["t5hmc"]).correlation),
        "mean_t5mc": float(cat["t5mc"].mean()),
        "mean_t5hmc": float(cat["t5hmc"].mean()),
        "seconds": time.time() - t0,
    }
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
