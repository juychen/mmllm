#!/usr/bin/env python
"""How much of the CpG evidence does a +/-N bp hole actually withhold?

Base-pair percentages of a 16 kb window are misleading: CpGs cluster, so a mask
that covers X% of the *span* can cover far more (or less) of the *CpG evidence*.
This walks the same test windows and, for each hole width, generates the real
seeded pair mask and counts how many CpGs end up hidden -- exact coverage, not
expectation.
"""
import argparse

import numpy as np
import torch

import run_xai_modality_shapley as X
from data import find_forward_cpg_positions, generate_cpg_pair_mask

BPS = (0, 10, 25, 50, 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--region-bed", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-windows", type=int, default=0)
    opts = ap.parse_args()
    opts.test_regions_bed = None
    opts.seed = None
    opts.mask_fraction = None

    ctx = X.prepare(opts)
    merged = ctx["merged"]
    indices = ctx["split_idx"]["test"]
    if opts.region_bed:
        hit, _ = X.region_bed_mask(ctx["split_df"], indices, opts.region_bed, "overlap")
        indices = indices[hit]
    if opts.max_windows:
        indices = X.sample_window_indices(indices, opts.max_windows, 7)
    print(f"[coverage] {len(indices)} windows", flush=True)

    _, loader = X.make_loader(merged, ctx["split_df"], indices, ctx["track_order"],
                              opts.batch_size, opts.num_workers)
    n_tracks = len(ctx["track_order"])
    mf = float(merged["mask_fraction"])
    base_seed = int(merged["seed"]) + 2000

    cpgs, wins = [], []
    hid_frac = {b: [] for b in BPS}
    scored_frac = {b: [] for b in BPS}
    span_hid = {b: [] for b in BPS}
    g = 0
    with torch.no_grad():
        for batch in loader:
            _, _, base_ids = X.split_batch(batch, n_tracks)
            for i in range(base_ids.shape[0]):
                one = base_ids[i:i + 1]
                pos = np.nonzero(find_forward_cpg_positions(one)[0].numpy())[0]
                if pos.size < 2:
                    continue
                cpgs.append(pos.size)
                wins.append(int((one != 0).sum()))
                for b in BPS:
                    hide, loss = generate_cpg_pair_mask(one, mf, b, seed=base_seed + g + i)
                    hide = hide[0, :, 0].bool().numpy()
                    sel = np.zeros_like(hide)
                    sel[pos] = True
                    hid_frac[b].append(float((hide & sel).sum()) / pos.size)
                    scored_frac[b].append(float((hide & sel).sum()) / pos.size)
                    span_hid[b].append(float(hide.sum()) / hide.size)
            g += base_ids.shape[0]

    cpgs = np.asarray(cpgs, float)
    wins = np.asarray(wins, float)
    print(f"windows={len(cpgs)}  CpG/window: mean={cpgs.mean():.0f} median={np.median(cpgs):.0f}"
          f"  window bp: mean={wins.mean():.0f}")
    print()
    print(f"{'hole':>5s} {'span_hidden%':>13s} {'CpG_hidden%':>12s}")
    for b in BPS:
        print(f"{b:5d} {100*np.mean(span_hid[b]):13.2f} {100*np.mean(hid_frac[b]):12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
