#!/usr/bin/env python3
"""Zero-shot 15% masked reconstruction on GSE244251 mouse Nanopore 5mC/5hmC.

Loads a trained 2-head MultiQueryMaskedPretrainingModel (query: 5mc, 5hmc;
context: atac, rna), feeds GSE244251 5mC/5hmC tracks, drops ATAC/RNA 100%
(the public dataset has no ATAC/RNA), masks 15% of covered CG sites, and
reports R2 / Pearson / Spearman at the masked CG sites.  No finetuning.

Important: the mask is the *shared* position mask used in pretraining, i.e. at
a masked CG site both 5mC and 5hmC hidden states are replaced by their
[POS-MASK] tokens.  The 5hmC head therefore uses neighbouring 5mC + sequence,
not the same-site 5mC.  This is the in-distribution masked-pretraining task.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import pyfaidx
import torch
from scipy.stats import spearmanr

from models import MultiQueryMaskedPretrainingModel

CELL_TYPES = ["Neuron", "Astrocyte", "Microglia"]
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def tabix_fetch(path: str, chrom: str, start: int, end: int):
    """Fetch 0-based [start, end) rows from a tabix-indexed bedGraph."""
    region = f"{chrom}:{start + 1}-{end}"
    proc = subprocess.run(["tabix", path, region], capture_output=True, text=True)
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


def load_track_window(path: str, chrom: str, start: int, length: int):
    """Return (values, covered) arrays of shape (length,), values in 0-100."""
    arr = np.zeros(length, dtype=np.float32)
    covered = np.zeros(length, dtype=bool)
    for s, e, v in tabix_fetch(path, chrom, start, start + length):
        a = max(0, s - start)
        b = min(length, e - start)
        if b > a:
            arr[a:b] = float(v) * 100.0
            covered[a:b] = True
    return arr, covered


def one_hot_sequence(seq: str):
    base = np.full(len(seq), 4, dtype=np.int64)
    for b, i in BASE_MAP.items():
        base[np.array(list(seq)) == b] = i
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i in range(4):
        x[base == i, i] = 1.0
    return x, base


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size < 2:
        return float("nan")
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pearson(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size < 2:
        return float("nan")
    x = pred - pred.mean()
    y = target - target.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / max(denom, 1e-12))


def spearman(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size < 2:
        return float("nan")
    return float(spearmanr(pred, target).correlation)


def linear_baseline_r2(x_fit, y_fit, x_eval, y_eval) -> float:
    """Fit y = a + b*x on unmasked sites, evaluate at masked sites."""
    if x_fit.size < 2 or np.std(x_fit) < 1e-8:
        return 0.0
    b, a = np.polyfit(x_fit, y_fit, 1)
    pred = a + b * x_eval
    return r2_score(pred, y_eval)


def load_regions_bed(path: str, chroms: list[str]):
    """Read BED regions; returns [(chrom, start, end), ...]."""
    if not path or not os.path.exists(path):
        return []
    opener = gzip.open if path.endswith(".gz") else open
    allowed = set(chroms)
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 3:
                continue
            chrom = f[0]
            # genome FASTA keys may be "1" while BED/tabix use "chr1"
            if chrom in allowed:
                genome_chrom = chrom
            elif chrom.startswith("chr") and chrom[3:] in allowed:
                genome_chrom = chrom[3:]
            elif ("chr" + chrom) in allowed:
                genome_chrom = "chr" + chrom
            else:
                continue
            data_chrom = chrom if chrom.startswith("chr") else "chr" + chrom
            try:
                start, end = int(float(f[1])), int(float(f[2]))
            except ValueError:
                continue
            if end > start:
                rows.append((genome_chrom, data_chrom, start, end))
    return rows


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    query_tracks = a.get("query_tracks") or ["5mc", "5hmc"]
    if list(query_tracks) != ["5mc", "5hmc"]:
        raise ValueError(f"expected query_tracks ['5mc','5hmc'], got {query_tracks}")
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
    ap.add_argument("--checkpoint", default="output/AMY_MC/cCRE_cpg/2026-10-20-13-31-30_multiquery_pretrain_ctxall-others_s0.0_a0.2_r0.2_best_80137.pt")
    ap.add_argument("--data-dir", default="/data1st1/junyi/methdata/GSE244251/processed_data_tabix")
    ap.add_argument("--genome", default="/data1st2/hannan_25/genome_ref/mm10/Mus_musculus.GRCm38.dna.primary_assembly.fa")
    ap.add_argument("--out-dir", default="output/GSE244251_zeroshot")
    ap.add_argument("--cell-types", nargs="+", default=CELL_TYPES, choices=CELL_TYPES)
    ap.add_argument("--max-windows", type=int, default=300, help="per cell type")
    ap.add_argument("--min-cg", type=int, default=20, help="minimum covered CG sites per 16 kb window")
    ap.add_argument("--regions-bed", default="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed",
                    help="candidate window BED; empty string -> random genome windows")
    ap.add_argument("--mask-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not shutil.which("tabix"):
        raise SystemExit("tabix not found in PATH")
    for p in [args.checkpoint, args.genome]:
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    model, ckpt_args, seq_len = load_model(args.checkpoint, device)
    genome = pyfaidx.Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    chroms = []
    for c in genome.keys():
        if c in ("chrM", "chrMT") or "random" in c.lower() or c.lower().endswith("_alt"):
            continue
        if len(genome[c]) > seq_len + 1000:
            chroms.append(c)
    if not chroms:
        raise SystemExit("no usable chromosomes in genome FASTA")
    regions = load_regions_bed(args.regions_bed, chroms)
    print(f"[regions] {len(regions)} candidate windows from {args.regions_bed or 'random genome'}")

    print(f"[model] {args.checkpoint}")
    print(f"[model] seq_len={seq_len} hidden={ckpt_args.get('hidden_dim')} blocks={ckpt_args.get('num_blocks')} "
          f"context={ckpt_args.get('context_mode')} | ATAC/RNA dropped 100%")
    print(f"[data] {args.data_dir} | cells={args.cell_types} | windows/cell={args.max_windows} "
          f"mask={args.mask_fraction} | device={device}")

    t0 = time.time()
    per_cell = {}
    all_masked = {"5mc": [], "5hmc": []}
    all_targets = {"5mc": [], "5hmc": []}

    for cell in args.cell_types:
        m5c_path = os.path.join(args.data_dir, f"5mC_{cell}_5mC.bedgraph.gz")
        h5c_path = os.path.join(args.data_dir, f"5hmC_{cell}_5hmC.bedgraph.gz")
        if not (os.path.exists(m5c_path) and os.path.exists(h5c_path)):
            print(f"[skip] {cell}: missing {m5c_path} / {h5c_path}")
            continue

        rng = np.random.default_rng(args.seed)
        masked = {k: [] for k in ("5mc", "5hmc")}
        unmasked = {k: [] for k in ("5mc", "5hmc")}
        cell_pred = {"5mc": [], "5hmc": []}
        n_windows = 0
        n_tried = 0
        if regions:
            order = rng.permutation(len(regions))
            candidates = (regions[i] for i in order)
            max_try = min(len(regions), args.max_windows * 10)
        else:
            candidates = None
            max_try = args.max_windows * 50

        for _ in range(max_try):
            if n_windows >= args.max_windows:
                break
            n_tried += 1
            if regions:
                chrom, data_chrom, rs, re = next(candidates)
                center = (rs + re) // 2
                start = max(0, center - seq_len // 2)
                if start + seq_len > len(genome[chrom]):
                    start = max(0, len(genome[chrom]) - seq_len)
            else:
                chrom = chroms[int(rng.integers(0, len(chroms)))]
                data_chrom = chrom if chrom.startswith("chr") else "chr" + chrom
                start = int(rng.integers(0, max(1, len(genome[chrom]) - seq_len + 1)))
            seq = str(genome[chrom][start:start + seq_len])
            if len(seq) != seq_len:
                continue

            seq_x, base = one_hot_sequence(seq)
            if float((base == 4).mean()) > 0.2:
                continue

            m5c_vals, cov_m = load_track_window(m5c_path, data_chrom, start, seq_len)
            h5c_vals, cov_h = load_track_window(h5c_path, data_chrom, start, seq_len)
            covered = cov_m & cov_h
            # CG C only: C followed by G
            cpg = (base == 1) & (np.roll(base, -1) == 2)
            cpg[-1] = False
            covered = covered & cpg
            n_cov = int(covered.sum())
            if n_cov < args.min_cg:
                continue

            n_mask = max(1, int(round(args.mask_fraction * n_cov)))
            idx = rng.choice(np.flatnonzero(covered), size=n_mask, replace=False)
            pos_mask = np.zeros((1, seq_len, 1), dtype=np.float32)
            pos_mask[0, idx, 0] = 1.0

            unmasked_idx = covered.copy()
            unmasked_idx[idx] = False
            for name, vals in (("5mc", m5c_vals), ("5hmc", h5c_vals)):
                masked[name].append(vals[idx])
                unmasked[name].append(vals[unmasked_idx])

            q5mc = torch.from_numpy(m5c_vals).view(1, seq_len, 1).to(device)
            q5hmc = torch.from_numpy(h5c_vals).view(1, seq_len, 1).to(device)
            ctx = [torch.zeros_like(q5mc), torch.zeros_like(q5mc)]
            seq_t = torch.from_numpy(seq_x).unsqueeze(0).to(device)
            pos_t = torch.from_numpy(pos_mask).to(device)
            qp = [torch.ones(1, 1, 1, device=device), torch.ones(1, 1, 1, device=device)]
            tp = [torch.zeros(1, 1, 1, device=device), torch.zeros(1, 1, 1, device=device)]
            amp_dtype = torch.bfloat16 if args.amp and device.type == "cuda" else torch.float32

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(args.amp and device.type == "cuda")):
                preds = model(
                    [q5mc, q5hmc], seq_t, ctx,
                    position_mask=pos_t,
                    query_present=qp,
                    track_present=tp,
                )
            pred_5mc = preds[0][0, :, 0].float().cpu().numpy()
            pred_5hmc = preds[1][0, :, 0].float().cpu().numpy()
            cell_pred["5mc"].append(pred_5mc[idx])
            cell_pred["5hmc"].append(pred_5hmc[idx])
            all_masked["5mc"].append(pred_5mc[idx])
            all_masked["5hmc"].append(pred_5hmc[idx])
            all_targets["5mc"].append(m5c_vals[idx])
            all_targets["5hmc"].append(h5c_vals[idx])
            n_windows += 1

        if n_windows == 0:
            print(f"[skip] {cell}: no window with >= {args.min_cg} covered CG sites")
            continue

        def cat(d, k):
            return np.concatenate(d[k]) if d[k] else np.zeros(0, dtype=np.float32)

        tgt5mc, tgt5hmc = cat(masked, "5mc"), cat(masked, "5hmc")
        n_cell = tgt5mc.size
        pred5mc = np.concatenate(cell_pred["5mc"]) if cell_pred["5mc"] else np.zeros(0, dtype=np.float32)
        pred5hmc = np.concatenate(cell_pred["5hmc"]) if cell_pred["5hmc"] else np.zeros(0, dtype=np.float32)
        fit_x = cat(unmasked, "5mc")
        fit_y = cat(unmasked, "5hmc")
        base_r2 = linear_baseline_r2(fit_x, fit_y, tgt5mc, tgt5hmc)

        row = {
            "n_windows": n_windows,
            "n_candidate_windows": n_tried,
            "n_masked_sites": int(n_cell),
            "5mc": {
                "r2": r2_score(pred5mc, tgt5mc),
                "pearson": pearson(pred5mc, tgt5mc),
                "spearman": spearman(pred5mc, tgt5mc),
            },
            "5hmc": {
                "r2": r2_score(pred5hmc, tgt5hmc),
                "pearson": pearson(pred5hmc, tgt5hmc),
                "spearman": spearman(pred5hmc, tgt5hmc),
                "mse": float(np.mean((pred5hmc - tgt5hmc) ** 2)),
                "mae": float(np.mean(np.abs(pred5hmc - tgt5hmc))),
                "target_mean": float(tgt5hmc.mean()),
                "target_std": float(tgt5hmc.std()),
                "linear_5mc_to_5hmc_r2": base_r2,
            },
        }
        per_cell[cell] = row
        print(f"[{cell}] windows={n_windows} sites={n_cell} | "
              f"5mc R2={row['5mc']['r2']:.4f} ρ={row['5mc']['pearson']:.4f} | "
              f"5hmc R2={row['5hmc']['r2']:.4f} ρ={row['5hmc']['pearson']:.4f} "
              f"(linear 5mC baseline R2={base_r2:.4f})")

    # pooled metrics
    pooled = {}
    for k in ("5mc", "5hmc"):
        if all_targets[k]:
            p = np.concatenate(all_masked[k])
            t = np.concatenate(all_targets[k])
            pooled[k] = {
                "r2": r2_score(p, t),
                "pearson": pearson(p, t),
                "spearman": spearman(p, t),
            }
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "genome": args.genome,
        "settings": {
            "cell_types": args.cell_types,
            "max_windows_per_cell": args.max_windows,
            "min_cg": args.min_cg,
            "mask_fraction": args.mask_fraction,
            "atac_drop_p": 1.0,
            "rna_drop_p": 1.0,
            "mask": "shared 15% of covered CG sites",
            "seq_len": seq_len,
            "finetuned": False,
        },
        "per_cell": per_cell,
        "pooled": pooled,
        "seconds": round(time.time() - t0, 1),
    }
    out_json = os.path.join(args.out_dir, f"GSE244251_zeroshot_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_json}")
    print(f"[pooled] 5mc R2={pooled.get('5mc', {}).get('r2', float('nan')):.4f} | "
          f"5hmc R2={pooled.get('5hmc', {}).get('r2', float('nan')):.4f}")


if __name__ == "__main__":
    main()
