#!/usr/bin/env python3
"""Paired window-bootstrap comparison of zero-shot arms on GSE279860.

Reports arm-level pooled R2 with CIs, robust per-window summaries (median R2,
fraction of catastrophic windows), and paired deltas. The pooled R2 is
heavy-tailed when an arm fails, so medians are reported alongside.
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
    return {k: d[k] for k in ("win", "pos", "t5mc", "p5mc", "t5hmc", "p5hmc")}


def r2(pred, target):
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def per_window_r2(d, p, t):
    out = []
    for w in np.unique(d["win"]):
        m = d["win"] == w
        if m.sum() < 5:
            continue
        out.append(r2(d[p][m], d[t][m]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data5/junyi/methdata/GSE279860_zs")
    ap.add_argument("--out-json", default="/data5/junyi/methdata/GSE279860_zs/bootstrap2.json")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    data = {f"{a}_bp{b}": load(os.path.join(args.dir, f"{a}_bp{b}.npz"))
            for a in ARMS for b in PROTS}
    ref = data["mixed_bp0"]
    keys = np.stack([ref["win"], ref["pos"]], axis=1)
    for k, d in data.items():
        k2 = np.stack([d["win"], d["pos"]], axis=1)
        if k2.shape != keys.shape or not np.array_equal(k2, keys):
            raise SystemExit(f"masked CpG sets differ for {k}")

    win = ref["win"]
    uniq = np.unique(win)
    idx = {w: np.flatnonzero(win == w) for w in uniq}
    rng = np.random.default_rng(args.seed)

    out = {"n_windows": int(uniq.size), "n_cpg": int(win.size), "arms": {}, "deltas": {}}
    for k, d in data.items():
        pw_mc = per_window_r2(d, "p5mc", "t5mc")
        pw_hm = per_window_r2(d, "p5hmc", "t5hmc")
        boots = {"5mc": [], "5hmc": []}
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            rows = np.concatenate([idx[w] for w in pick])
            boots["5mc"].append(r2(d["p5mc"][rows], d["t5mc"][rows]))
            boots["5hmc"].append(r2(d["p5hmc"][rows], d["t5hmc"][rows]))
        out["arms"][k] = {
            "r2_5mc": r2(d["p5mc"], d["t5mc"]),
            "r2_5hmc": r2(d["p5hmc"], d["t5hmc"]),
            "ci5mc": [float(np.percentile(boots["5mc"], 2.5)), float(np.percentile(boots["5mc"], 97.5))],
            "ci5hmc": [float(np.percentile(boots["5hmc"], 2.5)), float(np.percentile(boots["5hmc"], 97.5))],
            "median_pw_r2_5mc": float(np.median(pw_mc)),
            "median_pw_r2_5hmc": float(np.median(pw_hm)),
            "frac_pw_neg_5mc": float(np.mean(pw_mc < 0)),
            "frac_pw_neg_5hmc": float(np.mean(pw_hm < 0)),
        }

    for bp in PROTS:
        for a, b in [("mixed", "bp0"), ("mixed", "bp50"), ("bp50", "bp0")]:
            ka, kb = f"{a}_bp{bp}", f"{b}_bp{bp}"
            da, db = data[ka], data[kb]
            ds = {"5mc": [], "5hmc": []}
            for _ in range(args.n_boot):
                pick = rng.choice(uniq, size=uniq.size, replace=True)
                rows = np.concatenate([idx[w] for w in pick])
                for mod, t, p in (("5mc", "t5mc", "p5mc"), ("5hmc", "t5hmc", "p5hmc")):
                    ds[mod].append(r2(da[p][rows], da[t][rows]) - r2(db[p][rows], db[t][rows]))
            out["deltas"][f"{a}-{b}_bp{bp}"] = {
                mod: {
                    "delta_point": out["arms"][ka][f"r2_{mod}"] - out["arms"][kb][f"r2_{mod}"],
                    "delta_median": float(np.median(ds[mod])),
                    "ci_lo": float(np.percentile(ds[mod], 2.5)),
                    "ci_hi": float(np.percentile(ds[mod], 97.5)),
                    "frac_le_0": float(np.mean(np.array(ds[mod]) <= 0)),
                }
                for mod in ("5mc", "5hmc")
            }

    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"{'arm':>12} {'R2_5mc (CI)':>28} {'R2_5hmc (CI)':>28} {'pw<0 mc/hm':>12}")
    for k, v in out["arms"].items():
        print(f"{k:>12} {v['r2_5mc']:>7.4f} [{v['ci5mc'][0]:+.3f},{v['ci5mc'][1]:+.3f}] "
              f"{v['r2_5hmc']:>9.4f} [{v['ci5hmc'][0]:+.3f},{v['ci5hmc'][1]:+.3f}] "
              f"{v['frac_pw_neg_5mc']:.2f}/{v['frac_pw_neg_5hmc']:.2f}")
    print("\nDeltas (median of paired bootstrap):")
    for k, v in out["deltas"].items():
        print(f"{k:>16} 5mc {v['5mc']['delta_point']:+.4f} (med {v['5mc']['delta_median']:+.3f} "
              f"[{v['5mc']['ci_lo']:+.3f},{v['5mc']['ci_hi']:+.3f}])   "
              f"5hmc {v['5hmc']['delta_point']:+.4f} (med {v['5hmc']['delta_median']:+.3f} "
              f"[{v['5hmc']['ci_lo']:+.3f},{v['5hmc']['ci_hi']:+.3f}])")


if __name__ == "__main__":
    main()
