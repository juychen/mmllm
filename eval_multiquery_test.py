#!/usr/bin/env python
"""Evaluate ONE MultiQueryMaskedPretrainingModel checkpoint on its TEST set.

Reuses the training script's own data pipeline + evaluate(), so the numbers are
directly comparable to the run's printed "FINAL TEST METRICS".

  python eval_multiquery_test.py --checkpoint <..._best_*.pt> [--device N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_multiquery_pretraining as MQ            # noqa: E402
from models import MultiQueryMaskedPretrainingModel  # noqa: E402
from utils import derive_test_h5ad_path, export_multiquery_predictions_h5ad, get_freest_gpu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--bed-out-dir", default="/tmp/opencode/eval_beds",
                    help="where prepare_data writes its region BEDs (kept out of the run dir)")
    ap.add_argument("--export-h5ad", action=argparse.BooleanOptionalAction, default=True,
                    help="Export TEST predictions/targets/mask to .h5ad (default: on).")
    ap.add_argument("--h5ad-path", default="",
                    help="Output path (or template with {sample_size}/{timestamp}); empty derives it from "
                         "the checkpoint by replacing '_best_' with '_test_pred_' and '.pt' with '.h5ad'.")
    ap.add_argument("--h5ad-positions", choices=["cpg", "cpg_forward", "c"], default="cpg",
                    help="Positions stored in the .h5ad: 'cpg' (C and G of every CpG; matches the masked "
                         "metric, default), 'cpg_forward' (forward-strand C only) or 'c' (all C).")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ck["args"])
    num = int(ck["num_dmrs"])

    # keep prepare_data's BED side-effects away from the original run artifacts
    Path(a.bed_out_dir).mkdir(parents=True, exist_ok=True)
    args.output_csv = str(Path(a.bed_out_dir) / "eval_regions.csv")
    args.timestamp = str(args.timestamp) + "_eval"

    query_tracks, context_tracks = MQ.resolve_track_layout(args)
    prepared = MQ.prepare_data(num, args, query_tracks + context_tracks)

    if torch.cuda.is_available():
        if a.device is not None:
            dev = torch.device(f"cuda:{a.device}")
        elif os.environ.get("CUDA_VISIBLE_DEVICES"):
            dev = torch.device("cuda:0")          # only the pinned device is visible
        else:
            dev = torch.device(f"cuda:{get_freest_gpu()}")
    else:
        dev = torch.device("cpu")

    model = MultiQueryMaskedPretrainingModel(
        seq_len=prepared.seq_len,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=args.use_positional_encoding,
        num_blocks=args.num_blocks,
        fusion_type=args.fusion_type,
        num_queries=len(query_tracks),
        num_context_tracks=len(context_tracks),
        context_mode=getattr(args, "context_mode", "shared"),
    ).to(dev)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()

    split = getattr(prepared, "split_info", {})
    print(f"checkpoint      : {a.checkpoint}")
    print(f"  epoch         : {ck.get('epoch')}")
    print(f"  context_mode  : {getattr(args,'context_mode','shared')}")
    print(f"  dmr-csv       : {args.dmr_csv}")
    print(f"  test_bed      : {getattr(args,'test_bed',None)}")
    print(f"  split_case    : {split.get('split_case')}  "
          f"test_windows={prepared.test_regions}")

    if prepared.test_loader is None:
        raise SystemExit("no test set for this checkpoint (test_regions=0)")

    want_h5ad = a.export_h5ad
    if want_h5ad:
        m, payload = MQ.evaluate(
            model, prepared.test_loader, dev, args.mask_fraction,
            query_tracks, context_tracks, amp=args.amp, mask_seed=args.seed + 2000,
            return_predictions=True,
        )
    else:
        m = MQ.evaluate(model, prepared.test_loader, dev, args.mask_fraction,
                        query_tracks, context_tracks, amp=args.amp, mask_seed=args.seed + 2000)
    print("  --- TEST ---")
    for k in sorted(m):
        print(f"  {k:24s} = {m[k]:.4f}")

    if want_h5ad:
        h5ad_path = a.h5ad_path or derive_test_h5ad_path(a.checkpoint)
        h5ad_path = h5ad_path.format(sample_size=num, timestamp=getattr(args, "timestamp", ""))
        export_multiquery_predictions_h5ad(
            h5ad_path,
            payload["region_metadata"],
            payload["base_ids"],
            payload["predictions"],
            payload["targets"],
            payload["mask"],
            track_names=query_tracks,
            positions=a.h5ad_positions,
            uns_extra={
                "split": "test",
                "mask_fraction": args.mask_fraction,
                "test_r2": {n: m[f"val_{n}_r2"] for n in query_tracks},
                "test_pearsonr": {n: m[f"val_{n}_pearsonr"] for n in query_tracks},
                "checkpoint": str(a.checkpoint),
            },
        )
        print(f"  predictions   : {h5ad_path}")


if __name__ == "__main__":
    main()
