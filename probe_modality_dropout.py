#!/usr/bin/env python
"""Modality-absence sweep at inference time.

Training draws per-sample presence with ``--drop-p-*``; ``evaluate`` never does,
so every reported R^2 assumes all tracks present.  This probe reproduces the
*training-time* absence mechanism (hidden state replaced by the learned
[MODALITY-MASK] token via ``query_present`` / ``track_present`` /
``sequence_present``) for a deterministic subset of windows, and reports pooled
R^2 on the same seeded position mask.
"""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace

import torch

import run_multiquery_pretraining as R


def load_run(prefix: str):
    with open(f"{prefix}_results.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    return Namespace(**meta["args"]), int(meta["results"][0]["num_dmrs"])


class PresenceSampler:
    """Deterministic per-window presence flags for one absence mode."""

    def __init__(self, mode: str, p: float, num_q: int, num_c: int, device, seed: int = 1234):
        self.mode, self.p, self.num_q, self.num_c, self.device = mode, p, num_q, num_c, device
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, n: int):
        absent = (torch.rand(n, generator=self.g) < self.p)
        def flag(on: bool):
            v = torch.zeros(n) if on else absent.float()
            return v.view(n, 1, 1).to(self.device)
        qp = [flag(False) for _ in range(self.num_q)]
        tp = [flag(False) for _ in range(self.num_c)]
        sp = flag(False)
        if self.mode == "5mc":
            qp[0] = flag(True)
        elif self.mode == "atac":
            tp[0] = flag(True)
        elif self.mode == "rna":
            tp[-1] = flag(True)
        elif self.mode == "ctx":
            tp = [flag(True) for _ in range(self.num_c)]
        elif self.mode == "seq":
            sp = flag(True)
        elif self.mode != "base":
            raise ValueError(self.mode)
        return {"query_present": qp, "track_present": tp, "sequence_present": sp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", action="append", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="logs/probe_dropout.jsonl")
    ap.add_argument("--settings", default="base:0,ctx:0.5,ctx:1.0,5mc:1.0,seq:1.0")
    opts = ap.parse_args()

    device = torch.device(opts.device)
    settings = [(m, float(p)) for m, p in (s.split(":") for s in opts.settings.split(","))]
    out = open(opts.out, "a", encoding="utf-8")

    for prefix in opts.prefix:
        args, num_dmrs = load_run(prefix)
        query_tracks, context_tracks = R.resolve_track_layout(args)
        prepared = R.prepare_data(num_dmrs, args, query_tracks + context_tracks)
        model = R.MultiQueryMaskedPretrainingModel(
            seq_len=prepared.seq_len, hidden_dim=args.hidden_dim,
            use_positional_encoding=args.use_positional_encoding, num_blocks=args.num_blocks,
            fusion_type=args.fusion_type, num_queries=len(query_tracks),
            num_context_tracks=len(context_tracks), context_mode=args.context_mode,
        ).to(device)
        epoch, _ = R.load_model_from_checkpoint(model, f"{prefix}_best_{num_dmrs}.pt", device)
        tag = prefix.rstrip("/").split("/")[-1]
        print(f"[probe] {tag} ep{epoch} train drop_p atac/rna={args.drop_p_atac}/{args.drop_p_rna} seq={args.drop_p_seq}", flush=True)

        for split in [s for s in opts.splits.split(",") if s]:
            base_loader = prepared.val_loader if split == "val" else prepared.test_loader
            seed = args.seed + (1000 if split == "val" else 2000)
            for mode, p in settings:
                sampler = None if p == 0 else PresenceSampler(
                    mode, p, len(query_tracks), len(context_tracks), device)
                t0 = time.time()
                m = R.evaluate(model, base_loader, device, args.mask_fraction, query_tracks,
                               context_tracks, amp=bool(args.amp), mask_seed=seed,
                               mask_neighborhood_bp=args.mask_neighborhood_bp,
                               loss_mask_mode="hide_cpg", presence_fn=sampler)
                row = {"run": tag, "epoch": epoch, "split": split, "mode": mode, "p": p,
                       "r2": {n: float(m[f"val_{n}_r2"]) for n in query_tracks},
                       "sec": round(time.time() - t0, 1)}
                print(f"  {split} {mode}@{p}: " + " ".join(f"{k}={v:.4f}" for k, v in row["r2"].items())
                      + f"  ({row['sec']}s)", flush=True)
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
    out.close()


if __name__ == "__main__":
    main()