#!/usr/bin/env python
"""Why does a bp50-trained model score so low when only single bases are hidden?

Collects (y, pred) pairs for one checkpoint under several masks and prints the
shape of the error, not just R2:

  std(pred)/std(y)   -> is the output collapsed onto a smooth regional prior?
  corr(y, pred)      -> does it still rank CpGs correctly?
  slope              -> regression of pred on y (1 = calibrated, <1 = shrunk)
  bias               -> mean(pred) - mean(y)
"""
import argparse

import numpy as np
import torch

import run_xai_modality_shapley as X
from data import (find_forward_cpg_positions, generate_cpg_pair_mask,
                  generate_pretraining_cpg_mask)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--region-bed", default=None)
    ap.add_argument("--bps", default="0,50")
    ap.add_argument("--with-single", action="store_true", default=True)
    ap.add_argument("--heads", default="5hmc,5mc")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--tag", default="")
    opts = ap.parse_args()
    opts.test_regions_bed = None
    opts.seed = None
    opts.mask_fraction = None

    bps = [int(b) for b in opts.bps.split(",") if b.strip()]
    regimes = ([("single", None)] if opts.with_single else []) + [(f"bp{b}", b) for b in bps]
    heads = [h.strip() for h in opts.heads.split(",") if h.strip()]

    ctx = X.prepare(opts)
    merged, model, device = ctx["merged"], ctx["model"], ctx["device"]
    indices = ctx["split_idx"]["test"]
    if opts.region_bed:
        hit, _ = X.region_bed_mask(ctx["split_df"], indices, opts.region_bed, "overlap")
        indices = indices[hit]
    indices = X.sample_window_indices(indices, opts.max_windows, 7)
    print(f"[pred-stats]{opts.tag} {len(indices)} windows | regimes={[r[0] for r in regimes]}", flush=True)

    _, loader = X.make_loader(merged, ctx["split_df"], indices, ctx["track_order"],
                              opts.batch_size, opts.num_workers)
    query, n_q, n_tracks = ctx["query_tracks"], len(ctx["query_tracks"]), len(ctx["track_order"])
    mf = float(merged["mask_fraction"])
    base_seed = int(merged["seed"]) + 2000

    acc = {(r, h): {"y": [], "p": []} for r, _ in regimes for h in heads}
    model.eval()
    g = 0
    with torch.no_grad():
        for batch in loader:
            tracks, sequence, base_ids = X.split_batch(batch, n_tracks)
            bsz = base_ids.shape[0]
            single = torch.stack([generate_pretraining_cpg_mask(base_ids[i:i + 1], mf,
                                                               seed=base_seed + g + i)[0]
                                  for i in range(bsz)])
            hidden = {"single": single}
            scored = {"single": single.squeeze(-1).bool()}
            for name, bp in regimes:
                if bp is None:
                    continue
                pair = [generate_cpg_pair_mask(base_ids[i:i + 1], mf, bp,
                                               seed=base_seed + g + i) for i in range(bsz)]
                hidden[name] = torch.cat([p[0] for p in pair], 0)
                scored[name] = torch.cat([p[1] for p in pair], 0).squeeze(-1).bool()
            fwd = find_forward_cpg_positions(base_ids).bool()
            for name, _ in regimes:
                scored[name] = scored[name] & fwd

            q_dev = [t.to(device) for t in tracks[:n_q]]
            c_dev = [t.to(device) for t in tracks[n_q:]]
            s_dev = sequence.to(device)
            for name, _ in regimes:
                preds = model(q_dev, s_dev, c_dev, position_mask=hidden[name].to(device))
                m = scored[name]
                for h in heads:
                    hi = query.index(h)
                    acc[(name, h)]["y"].append(tracks[hi].squeeze(-1)[m].numpy().astype(np.float64))
                    acc[(name, h)]["p"].append(preds[hi].squeeze(-1).float().cpu()[m].numpy().astype(np.float64))
            g += bsz

    print()
    print(f"{'mask':7s} {'head':5s} {'n':>7s} {'std_y':>7s} {'std_p':>7s} {'ratio':>6s} "
          f"{'corr':>6s} {'slope':>6s} {'bias':>7s} {'R2':>7s}")
    for name, _ in regimes:
        for h in heads:
            y = np.concatenate(acc[(name, h)]["y"])
            p = np.concatenate(acc[(name, h)]["p"])
            if y.size < 50:
                continue
            sy, sp = y.std(), p.std()
            A = np.vstack([y - y.mean(), np.ones_like(y)]).T
            coef, *_ = np.linalg.lstsq(A, p - p.mean(), rcond=None)
            r2 = 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            print(f"{name:7s} {h:5s} {y.size:7d} {sy:7.2f} {sp:7.2f} {sp/sy:6.2f} "
                  f"{np.corrcoef(y, p)[0, 1]:6.3f} {coef[0]:6.3f} {p.mean()-y.mean():7.2f} {r2:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
