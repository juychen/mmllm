#!/usr/bin/env python
"""Score a checkpoint twice: anchor-only (training) mask vs full hole completion.

For the same held-out windows and the same seeded position mask, we evaluate

  loss      : R^2 at the C of the 15% selected CpG pairs (what training scores)
  hide_cpg  : R^2 at EVERY forward C that falls inside the hole (deployment task)

so a run that only ever scored anchors is judged on the same set of positions as
a run that scored every hidden CpG.
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace

import torch

import run_multiquery_pretraining as R


def load_run(prefix: str) -> tuple[Namespace, int]:
    with open(f"{prefix}_results.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    args = Namespace(**meta["args"])
    num_dmrs = int(meta["results"][0]["num_dmrs"])
    return args, num_dmrs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="path without _results.json / _best_N.pt suffix")
    ap.add_argument("--ckpt", default="", help="checkpoint path (default: <prefix>_best_<num_dmrs>.pt)")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--base-prefix", default="", help="use this run's results.json when the target run has not written one yet")
    opts = ap.parse_args()

    if opts.base_prefix and not __import__("pathlib").Path(f"{opts.prefix}_results.json").exists():
        args, num_dmrs = load_run(opts.base_prefix)
    else:
        args, num_dmrs = load_run(opts.prefix)
    ckpt = opts.ckpt or f"{opts.prefix}_best_{num_dmrs}.pt"
    device = torch.device(opts.device)

    query_tracks, context_tracks = R.resolve_track_layout(args)
    prepared = R.prepare_data(num_dmrs, args, query_tracks + context_tracks)

    model = R.MultiQueryMaskedPretrainingModel(
        seq_len=prepared.seq_len,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=args.use_positional_encoding,
        num_blocks=args.num_blocks,
        fusion_type=args.fusion_type,
        num_queries=len(query_tracks),
        num_context_tracks=len(context_tracks),
        context_mode=args.context_mode,
    ).to(device)
    epoch, _ = R.load_model_from_checkpoint(model, ckpt, device)

    tag = opts.tag or opts.prefix.rstrip("/").split("/")[-1]
    print(f"[eval] {tag} | ckpt={ckpt} | epoch={epoch}")
    print(f"[eval] mask_fraction={args.mask_fraction} neighborhood=+/-{args.mask_neighborhood_bp} bp")

    for split in [s.strip() for s in opts.splits.split(",") if s.strip()]:
        loader = prepared.val_loader if split == "val" else prepared.test_loader
        if loader is None:
            print(f"[eval] {split}: no loader")
            continue
        seed = args.seed + (1000 if split == "val" else 2000)
        row = {"split": split}
        for mode in ("loss", "hide_cpg"):
            metrics = R.evaluate(
                model, loader, device, args.mask_fraction,
                query_tracks, context_tracks, amp=bool(args.amp),
                mask_seed=seed, mask_neighborhood_bp=args.mask_neighborhood_bp,
                loss_mask_mode=mode,
            )
            row[f"n_{mode}"] = metrics["eval_n_positions"]
            for name in query_tracks:
                row[f"{mode}_{name}_r2"] = round(metrics[f"val_{name}_r2"], 4)
                row[f"{mode}_{name}_rho"] = round(metrics[f"val_{name}_pearsonr"], 4)
            print(f"  {split:5s} {mode:8s} n={row[f'n_{mode}']:>8,} "
                  + "  ".join(f"{n}: R2={row[f'{mode}_{n}_r2']:.4f} rho={row[f'{mode}_{n}_rho']:.4f}"
                              for n in query_tracks))
    print("[done]")


if __name__ == "__main__":
    main()
