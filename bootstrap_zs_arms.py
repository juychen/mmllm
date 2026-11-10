#!/usr/bin/env python3
"""Paired window-bootstrap comparison of zero-shot arms on GSE279860.

Reads the per-CpG npz files produced by eval_external_zs.py (same windows and
same masked CpGs across arms/protocols) and reports pooled R2 plus bootstrap
CIs for arm differences, overall and by CpG spacing stratum.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ARMS = ["mixed", "bp50", "bp0"]
PROTS = [0, 50]


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in ("win", "pos", "spacing", "t5mc", "p5mc", "t5hmc", "p5hmc")}


def r2(pred, target):
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data5/junyi/methdata/GSE279860_zs")
    ap.add_argument("--out-json", default="/data5/junyi/methdata/GSE279860_zs/bootstrap.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    data = {}
    for arm in ARMS:
        for bp in PROTS:
            key = f"{arm}_bp{bp}"
            data[key] = load(os.path.join(args.dir, key + ".npz"))

    ref = data["mixed_bp0"]
    keys = np.stack([ref["win"], ref["pos"]], axis=1)
    for key, d in data.items():
        k2 = np.stack([d["win"], d["pos"]], axis=1)
        if k2.shape != keys.shape or not np.array_equal(k2, keys):
            raise SystemExit(f"masked CpG sets differ for {key}")

    win = ref["win"]
    uniq = np.unique(win)
    idx_by_win = {w: np.flatnonzero(win == w) for w in uniq}

    rng = np.random.default_rng(args.seed)
    out = {"n_windows": int(uniq.size), "n_cpg": int(win.size), "point": {}, "boot": {}, "strata": {}}

    for key, d in data.items():
        out["point"][key] = {
            "r2_5mc": r2(d["p5mc"], d["t5mc"]),
            "r2_5hmc": r2(d["p5hmc"], d["t5hmc"]),
        }

    pairs = [("mixed", "bp0"), ("mixed", "bp50"), ("bp50", "bp0")]
    for bp in PROTS:
        for a, b in pairs:
            ka, kb = f"{a}_bp{bp}", f"{b}_bp{bp}"
            da, db = data[ka], data[kb]
            deltas = {"5mc": [], "5hmc": []}
            for _ in range(args.n_boot):
                pick = rng.choice(uniq, size=uniq.size, replace=True)
                rows = np.concatenate([idx_by_win[w] for w in pick])
                for mod, t, p in (("5mc", "t5mc", "p5mc"), ("5hmc", "t5hmc", "p5hmc")):
                    deltas[mod].append(r2(da[p][rows], da[t][rows]) - r2(db[p][rows], db[t][rows]))
            out["boot"][f"{a}-{b}_bp{bp}"] = {
                mod: {
                    "delta": float(np.mean(deltas[mod])),
                    "ci_lo": float(np.percentile(deltas[mod], 2.5)),
                    "ci_hi": float(np.percentile(deltas[mod], 97.5)),
                    "p_le_0": float(np.mean(np.array(deltas[mod]) <= 0)),
                }
                for mod in ("5mc", "5hmc")
            }

    bins = [(0, 10), (11, 50), (51, 100), (101, 10**9)]
    labels = ["<=10bp", "11-50bp", "51-100bp", ">100bp"]
    for key, d in data.items():
        sp = d["spacing"]
        out["strata"][key] = {}
        for (lo, hi), lab in zip(bins, labels):
            m = (sp >= lo) & (sp <= hi)
            if m.sum() < 50:
                continue
            out["strata"][key][lab] = {
                "n": int(m.sum()),
                "r2_5mc": r2(d["p5mc"][m], d["t5mc"][m]),
                "r2_5hmc": r2(d["p5hmc"][m], d["t5hmc"][m]),
            }

    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"{'arm':>10} {'mask':>5} {'R2_5mc':>8} {'R2_5hmc':>9} {'n_cpg':>7}")
    for key in sorted(out["point"]):
        p = out["point"][key]
        print(f"{key:>10} {key.split('_bp')[1]:>5} {p['r2_5mc']:>8.4f} {p['r2_5hmc']:>9.4f} {out['n_cpg']:>7}")
    print("\nDelta (paired window bootstrap, 10k):")
    for k, v in out["boot"].items():
        print(f"{k:>16}  5mc {v['5mc']['delta']:+.4f} [{v['5mc']['ci_lo']:+.4f},{v['5mc']['ci_hi']:+.4f}]   "
              f"5hmc {v['5hmc']['delta']:+.4f} [{v['5hmc']['ci_lo']:+.4f},{v['5hmc']['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
