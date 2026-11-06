#!/usr/bin/env python
"""Score the SAME windows under a grid of masking regimes, for one checkpoint.

Masks
  single : legacy 15%-of-CpG-positions mask (one base hidden, no buffer) -- what
           run_xai_modality_shapley.py applied in the 11-04 DMR run
  bpN    : CpG-pair mask, hide the pair's C and G plus +/-N bp (neighbourhood_bp=N)
           bp0 is the "no neighbourhood" protocol, bp50 the checkpoint's own one

Scored positions are always the forward C of each masked CpG pair -- exactly the
positions the C-only loss is defined on, so R2 is comparable across masks in
*definition* (not in difficulty: a wider hole hides more information).

Use it to build a capability curve: one row per (checkpoint, mask, head).

Usage:
  python probe_mask_regimes.py \
      --checkpoint  output/AMY_MC/cCRE_cpg/<run>_best_80137.pt \
      --results-json output/AMY_MC/cCRE_cpg/<run>_xai_compat_results.json \
      --region-bed output/dmr_with_sequences.bed \
      --bps 0,10,25,50,100 --heads 5hmc,5mc --device cuda:6
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
    ap.add_argument("--bps", default="0,50", help="comma list of neighbourhood widths")
    ap.add_argument("--heads", default="5hmc,5mc")
    ap.add_argument("--with-single", action="store_true", default=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--iso-bp", type=int, default=0,
                    help="only score CpGs whose nearest other CpG is farther than this")
    opts = ap.parse_args()
    opts.test_regions_bed = None
    opts.seed = None
    opts.mask_fraction = None

    bps = [int(b) for b in str(opts.bps).split(",") if str(b).strip()]
    regimes = ([("single", None)] if opts.with_single else []) + [(f"bp{b}", b) for b in bps]

    ctx = X.prepare(opts)
    merged, model, device = ctx["merged"], ctx["model"], ctx["device"]
    indices = ctx["split_idx"]["test"]
    if opts.region_bed:
        hit, _ = X.region_bed_mask(ctx["split_df"], indices, opts.region_bed, "overlap")
        indices = indices[hit]
    if opts.max_windows:
        indices = X.sample_window_indices(indices, opts.max_windows, 7)
    print(f"[probe]{opts.tag} {len(indices)} windows | mask_fraction={merged['mask_fraction']} "
          f"| regimes={[r[0] for r in regimes]}", flush=True)

    _, loader = X.make_loader(merged, ctx["split_df"], indices, ctx["track_order"],
                              opts.batch_size, opts.num_workers)
    query = ctx["query_tracks"]
    n_q, n_tracks = len(query), len(ctx["track_order"])
    heads = [h.strip() for h in str(opts.heads).split(",") if h.strip()]
    mf = float(merged["mask_fraction"])
    base_seed = int(merged["seed"]) + 2000

    acc = {(r, h): {"sse": 0.0, "s": 0.0, "s2": 0.0, "n": 0, "pw": []}
           for r, _ in regimes for h in heads}
    model.eval()
    g = 0
    with torch.no_grad():
        for batch in loader:
            tracks, sequence, base_ids = X.split_batch(batch, n_tracks)
            bsz = base_ids.shape[0]
            single = torch.stack([
                generate_pretraining_cpg_mask(base_ids[i:i + 1], mf, seed=base_seed + g + i)[0]
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
            if opts.iso_bp > 0:
                iso = torch.zeros_like(fwd)
                for i in range(bsz):
                    pp = torch.nonzero(fwd[i]).flatten().tolist()
                    if len(pp) < 2:
                        continue
                    arr = np.asarray(pp)
                    dd = np.abs(arr[:, None] - arr[None, :])
                    np.fill_diagonal(dd, 10**9)
                    ok = arr[dd.min(axis=1) > opts.iso_bp]
                    iso[i, torch.from_numpy(ok).long()] = True
                fwd = iso
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
                    y = tracks[hi].squeeze(-1)[m].double()
                    p = preds[hi].squeeze(-1).float().cpu()[m].double()
                    st = acc[(name, h)]
                    st["sse"] += float(((y - p) ** 2).sum())
                    st["s"] += float(y.sum())
                    st["s2"] += float((y ** 2).sum())
                    st["n"] += int(y.numel())
            g += bsz
            if g % 480 == 0:
                print(f"  [{g}/{len(indices)}]", flush=True)

    print()
    print(f"{'mask':8s} {'head':6s} {'n_cpg':>8s} {'pooled_R2':>10s}")
    for name, _ in regimes:
        for h in heads:
            st = acc[(name, h)]
            n = st["n"]
            sst = st["s2"] - st["s"] ** 2 / n
            print(f"{name:8s} {h:6s} {n:8d} {1.0 - st['sse'] / sst:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
