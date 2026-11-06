#!/usr/bin/env python
"""How much of a hidden CpG's 5hmC can the nearest *visible* CpG explain?

Model-free baseline.  Under a bpN mask the neighbouring CpGs are hidden too, so
the nearest visible CpG moves farther away as N grows.  Fitting
    y_hidden = a * y_nearest_visible + b
gives the variance a naive "copy the neighbour" predictor would capture -- the
ceiling any amount of neighbour-copying can reach under that mask.
"""
import argparse

import numpy as np
import torch

import run_xai_modality_shapley as X
from data import find_forward_cpg_positions, generate_cpg_pair_mask


def r2(y, p):
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / sst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--region-bed", default=None)
    ap.add_argument("--bps", default="0,50")
    ap.add_argument("--heads", default="5hmc,5mc")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-windows", type=int, default=0)
    opts = ap.parse_args()
    opts.test_regions_bed = None
    opts.seed = None
    opts.mask_fraction = None

    bps = [int(b) for b in opts.bps.split(",") if b.strip()]
    heads = [h.strip() for h in opts.heads.split(",") if h.strip()]

    ctx = X.prepare(opts)
    merged = ctx["merged"]
    indices = ctx["split_idx"]["test"]
    if opts.region_bed:
        hit, _ = X.region_bed_mask(ctx["split_df"], indices, opts.region_bed, "overlap")
        indices = indices[hit]
    if opts.max_windows:
        indices = X.sample_window_indices(indices, opts.max_windows, 7)
    print(f"[copy-index] {len(indices)} windows", flush=True)

    _, loader = X.make_loader(merged, ctx["split_df"], indices, ctx["track_order"],
                              opts.batch_size, opts.num_workers)
    n_tracks, query = len(ctx["track_order"]), ctx["query_tracks"]
    mf = float(merged["mask_fraction"])
    base_seed = int(merged["seed"]) + 2000

    out = {(b, h): {"y": [], "nn": [], "dist": []} for b in bps for h in heads}
    g = 0
    with torch.no_grad():
        for batch in loader:
            tracks, _, base_ids = X.split_batch(batch, n_tracks)
            for i in range(base_ids.shape[0]):
                one = base_ids[i:i + 1]
                pos = np.nonzero(find_forward_cpg_positions(one)[0].numpy())[0]
                if pos.size < 3:
                    continue
                pid2idx = {int(p): k for k, p in enumerate(pos)}
                for b in bps:
                    hide, loss = generate_cpg_pair_mask(one, mf, b, seed=base_seed + g + i)
                    hidden = hide[0, :, 0].bool().numpy()
                    scored = loss[0, :, 0].bool().numpy() & hidden
                    visible = pos[~hidden[pos]]
                    if visible.size < 2:
                        continue
                    sel = pos[scored[pos]]
                    if sel.size == 0:
                        continue
                    for p in sel:
                        j = int(np.argmin(np.abs(visible - p)))
                        q, d = int(visible[j]), int(abs(visible[j] - p))
                        if d > 5000:
                            continue
                        for h in heads:
                            hi = query.index(h)
                            tr = tracks[hi][i, :, 0].float().numpy()
                            out[(b, h)]["y"].append(float(tr[p]))
                            out[(b, h)]["nn"].append(float(tr[q]))
                            out[(b, h)]["dist"].append(d)
            g += base_ids.shape[0]

    print()
    print(f"{'mask':6s} {'head':6s} {'n':>8s} {'med_dist':>9s} {'R2_nn':>7s} {'slope':>7s}")
    for b in bps:
        for h in heads:
            d = out[(b, h)]
            y = np.asarray(d["y"], float)
            x = np.asarray(d["nn"], float)
            if y.size < 50:
                continue
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            print(f"bp{b:<4d} {h:6s} {y.size:8d} {np.median(d['dist']):9.0f} "
                  f"{r2(y, pred):7.3f} {coef[0]:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
