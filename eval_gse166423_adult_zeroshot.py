#!/usr/bin/env python3
"""Zero-shot 15% masked evaluation on GSE166423 Adult Purkinje neurons.

Model: cCRE 2-head a0.2 (query 5mc/5hmc; context atac/rna).
Data: GSE166423 Adult scaled 5mC/5hmC (0-100), real ATAC (min-max per window),
      zero RNA (GSE166423 has no RNA; RNA is dropped).
Regions: cCRE held-out test BED (chr8/9/X/Y), by default.
No finetuning.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pyBigWig
import pyfaidx
import torch
from scipy.stats import spearmanr

from models import MultiQueryMaskedPretrainingModel

BASE = "/data1st1/junyi/methdata/GSE166423_finetune_input"
GENOME = "/data1st2/hannan_25/genome_ref/mm10/Mus_musculus.GRCm38.dna.primary_assembly.fa"
TESTBED = "/home/junyichen/code/mmllm/output/holdout_testchroms/cpg_test.bed"
CKPT = "output/AMY_MC/cCRE_cpg/2026-10-20-13-31-30_multiquery_pretrain_ctxall-others_s0.0_a0.2_r0.2_best_80137.pt"
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def find_chrom(chrom: str, genome):
    if chrom in genome:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in genome:
        return chrom[3:]
    if ("chr" + chrom) in genome:
        return "chr" + chrom
    return None


def load_bw(path: str, chrom: str, start: int, length: int):
    bw = pyBigWig.open(path)
    try:
        raw = np.asarray(bw.values(chrom, start, start + length), dtype=np.float64)
    finally:
        bw.close()
    cov = np.isfinite(raw)
    vals = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return vals, cov


def atac_minmax(raw: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def one_hot(seq: str):
    base = np.full(len(seq), 4, dtype=np.int64)
    arr = np.array(list(seq))
    for b, i in BASE_MAP.items():
        base[arr == b] = i
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i in range(4):
        x[base == i, i] = 1.0
    return x, base


def r2(pred, target):
    if pred.size < 2:
        return float("nan")
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pearson(pred, target):
    if pred.size < 2:
        return float("nan")
    x = pred - pred.mean(); y = target - target.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / max(den, 1e-12))


def spearman(pred, target):
    return float(spearmanr(pred, target).correlation) if pred.size >= 2 else float("nan")


def linear_baseline(x_fit, y_fit, x_eval, y_eval):
    if x_fit.size < 2 or np.std(x_fit) < 1e-8:
        return 0.0
    b, a = np.polyfit(x_fit, y_fit, 1)
    return r2(a + b * x_eval, y_eval)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
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
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, int(a.get("target_length", 16384))


def load_regions(path, genome):
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 3:
                continue
            g = find_chrom(f[0], genome)
            if g is None:
                continue
            try:
                s, e = int(float(f[1])), int(float(f[2]))
            except ValueError:
                continue
            if e > s:
                rows.append((g, f[0] if f[0].startswith("chr") else "chr" + g, s, e))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--m5c-bw", default=os.path.join(BASE, "Purkinje_Adult_5mC_scaled100.bw"))
    ap.add_argument("--hm5c-bw", default=os.path.join(BASE, "Purkinje_Adult_5hmC_scaled100.bw"))
    ap.add_argument("--atac-bw", default=os.path.join(BASE, "Purkinje_Adult_ATAC.bw"))
    ap.add_argument("--rna-bw", default=os.path.join(BASE, "zero_RNA_mm10.bw"))
    ap.add_argument("--regions-bed", default=TESTBED)
    ap.add_argument("--genome", default=GENOME)
    ap.add_argument("--out-dir", default="output/GSE166423_Adult_zeroshot")
    ap.add_argument("--max-windows", type=int, default=100)
    ap.add_argument("--min-cg", type=int, default=20)
    ap.add_argument("--mask-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, seq_len = load_model(args.checkpoint, device)
    genome = pyfaidx.Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    regions = load_regions(args.regions_bed, genome)
    if not regions:
        raise SystemExit(f"no usable regions in {args.regions_bed}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(regions))

    pred = {"5mc": [], "5hmc": []}
    tgt = {"5mc": [], "5hmc": []}
    unm = {"5mc": [], "5hmc": []}
    n_win = 0
    for ri in order:
        if n_win >= args.max_windows:
            break
        gchrom, dchrom, rs, re = regions[ri]
        center = (rs + re) // 2
        start = max(0, center - seq_len // 2)
        if start + seq_len > len(genome[gchrom]):
            start = max(0, len(genome[gchrom]) - seq_len)
        seq = str(genome[gchrom][start:start + seq_len])
        if len(seq) != seq_len:
            continue
        seq_x, base = one_hot(seq)
        if float((base == 4).mean()) > 0.2:
            continue

        m5c, cm = load_bw(args.m5c_bw, dchrom, start, seq_len)
        h5c, ch = load_bw(args.hm5c_bw, dchrom, start, seq_len)
        atac_raw, _ = load_bw(args.atac_bw, dchrom, start, seq_len)
        atac = atac_minmax(atac_raw)
        rna = np.zeros(seq_len, dtype=np.float32)

        cpg = (base == 1) & (np.roll(base, -1) == 2)
        cpg[-1] = False
        covered = cm & ch & cpg
        n_cov = int(covered.sum())
        if n_cov < args.min_cg:
            continue

        n_mask = max(1, int(round(args.mask_fraction * n_cov)))
        idx = rng.choice(np.flatnonzero(covered), size=n_mask, replace=False)
        pm = np.zeros((1, seq_len, 1), dtype=np.float32)
        pm[0, idx, 0] = 1.0
        keep = covered.copy(); keep[idx] = False
        for nm, vals in (("5mc", m5c), ("5hmc", h5c)):
            tgt[nm].append(vals[idx]); unm[nm].append(vals[keep])

        q5mc = torch.from_numpy(m5c.astype(np.float32)).view(1, seq_len, 1).to(device)
        q5hmc = torch.from_numpy(h5c.astype(np.float32)).view(1, seq_len, 1).to(device)
        qatac = torch.from_numpy(atac).view(1, seq_len, 1).to(device)
        qrna = torch.from_numpy(rna).view(1, seq_len, 1).to(device)
        seq_t = torch.from_numpy(seq_x).unsqueeze(0).to(device)
        pm_t = torch.from_numpy(pm).to(device)
        qp = [torch.ones(1, 1, 1, device=device), torch.ones(1, 1, 1, device=device)]
        tp = [torch.ones(1, 1, 1, device=device), torch.zeros(1, 1, 1, device=device)]  # ATAC present, RNA absent
        amp_dtype = torch.bfloat16 if args.amp and device.type == "cuda" else torch.float32
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(args.amp and device.type == "cuda")):
            out = model([q5mc, q5hmc], seq_t, [qatac, qrna],
                        position_mask=pm_t, query_present=qp, track_present=tp)
        pred["5mc"].append(out[0][0, :, 0].float().cpu().numpy()[idx])
        pred["5hmc"].append(out[1][0, :, 0].float().cpu().numpy()[idx])
        n_win += 1

    if n_win == 0:
        raise SystemExit("no window passed coverage filters")
    res = {}
    for nm in ("5mc", "5hmc"):
        p = np.concatenate(pred[nm]); t = np.concatenate(tgt[nm])
        ux = np.concatenate(unm["5mc"]); uy = np.concatenate(unm["5hmc"])
        res[nm] = {
            "n_masked_sites": int(t.size),
            "r2": r2(p, t),
            "pearson": pearson(p, t),
            "spearman": spearman(p, t),
            "mse": float(np.mean((p - t) ** 2)),
            "mae": float(np.mean(np.abs(p - t))),
            "target_mean": float(t.mean()),
            "target_std": float(t.std()),
        }
        if nm == "5hmc":
            res[nm]["linear_5mc_to_5hmc_r2"] = linear_baseline(ux, uy, np.concatenate(tgt["5mc"]), t)

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "checkpoint": args.checkpoint,
        "dataset": "GSE166423 Adult Purkinje",
        "settings": {
            "m5c": args.m5c_bw, "hm5c": args.hm5c_bw, "atac": args.atac_bw,
            "rna": "zero (track_present=0)", "regions": args.regions_bed,
            "max_windows": args.max_windows, "min_cg": args.min_cg,
            "mask_fraction": args.mask_fraction, "atac_scaling": "per-window minmax",
            "atac_drop_p": 0.0, "rna_drop_p": 1.0, "finetuned": False,
        },
        "n_windows": n_win,
        "metrics": res,
    }
    out_json = os.path.join(args.out_dir, f"GSE166423_Adult_zeroshot_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[windows={n_win}]")
    for nm in ("5mc", "5hmc"):
        r = res[nm]
        print(f"{nm}: n={r['n_masked_sites']} R2={r['r2']:.4f} rho={r['pearson']:.4f} "
              f"MSE={r['mse']:.2f} MAE={r['mae']:.2f}")
    print(f"5hmc linear 5mC baseline R2={res['5hmc'].get('linear_5mc_to_5hmc_r2', float('nan')):.4f}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
