#!/usr/bin/env python
"""Rebuild a run's results CSV/JSON from its checkpoint — no retraining.

Why this exists
---------------
``run_experiment()`` only builds its ``ExperimentResult`` AFTER training, the
held-out test evaluation, and the ``.h5ad`` / regression-plot export. If that
final assembly raises (e.g. the ``split_info`` NameError in commit 7fdeb22),
you keep the log, the checkpoints, the ``.h5ad`` and the plot — but lose
``results.csv`` / ``results.json``. This script re-derives that result:

  * ``best_*``           : read straight from the checkpoint's ``metrics``.
  * ``final_val_*`` / ``test_*`` : recomputed by loading the checkpoint's
    weights and re-running the module's OWN ``evaluate()`` on val/test. This is
    exact, not an approximation — ``run_experiment()`` also evaluates the BEST
    weights for these fields (it reloads ``best_state`` before the final val
    eval), and both loaders are deterministic (fixed split; the loss mask comes
    from the DNA sequence).

Everything is driven by the ``args`` dict stored INSIDE the checkpoint, so the
same split, model architecture and data pipeline are reconstructed.

Usage
-----
  python recover_results_from_checkpoint.py \
      --run-script run_m5c_query_crosshyena_dropmod.py \
      --checkpoint output/ALL_GROUPS/cCRE_cpg/2026-10-07-16-10-41_...._best_480822.pt

  # optional:
  #   --out-csv / --out-json   override the output paths
  #   --last-checkpoint P.pt   (only used to recover final_lr exactly)
  #   --device 3               pin a GPU (default: freest GPU)
  #   --dry-run                evaluate + print, write nothing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import models as M
import models_sequence_only as MSO
from utils import get_freest_gpu


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_run_module(path: str):
    """Import a training script by file path (its dir is prepended to sys.path)."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"--run-script not found: {p}")
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(f"_recover_{p.stem}", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_prepare_fn(module):
    """The module's single ``prepare_*_data`` function."""
    names = [n for n in dir(module) if n.startswith("prepare_") and n.endswith("_data") and callable(getattr(module, n))]
    if len(names) != 1:
        raise RuntimeError(f"expected exactly one prepare_*_data in module, found {names}")
    return getattr(module, names[0])


def build_model(module, args, prepared, state_dict, device):
    """Rebuild the exact architecture from the checkpoint's state_dict keys."""
    keys = set(state_dict)
    hidden = args.hidden_dim
    seq_len = prepared.seq_len
    nblocks = getattr(args, "model_b_blocks", getattr(args, "num_blocks", 2))
    fusion = getattr(args, "model_b_fusion", "cross_hyena")
    pe = getattr(args, "use_positional_encoding", False)

    is_sequence_only = hasattr(module, "M5CQuerySequenceOnlyCrossHyenaRegressorModelB") or "M5CQuerySequenceOnly" in dir(module)

    if any(k.startswith("backbone.") for k in keys):                     # MinimalCrossHyenaRegressor
        return M.MinimalCrossHyenaRegressor(
            seq_len=seq_len, query_dim=1, context_dim=4, hidden_dim=hidden,
            post_filter_len=prepared.post_filter_len, use_positional_encoding=pe,
        ).to(device)
    if "rna_proj.weight" in keys:                                        # ModelB + RNA
        return M.M5CQuerySequenceAtacRnaCrossHyenaRegressorModelB(
            seq_len=seq_len, hidden_dim=hidden, use_positional_encoding=pe,
            num_blocks=nblocks, fusion_type=fusion, rna_dim=1,
        ).to(device)
    if "atac_proj.weight" in keys:
        if is_sequence_only:                                             # sequence-only model, ATAC branch
            return MSO.M5CQuerySequenceOnlyCrossHyenaRegressorModelB(
                seq_len=seq_len, hidden_dim=hidden, use_positional_encoding=pe,
                num_blocks=nblocks, fusion_type=fusion, use_atac=True,
            ).to(device)
        return M.M5CQuerySequenceAtacCrossHyenaRegressorModelB(          # ModelB + ATAC
            seq_len=seq_len, hidden_dim=hidden, use_positional_encoding=pe,
            num_blocks=nblocks, fusion_type=fusion,
        ).to(device)
    if is_sequence_only:                                                 # sequence-only, no ATAC
        return MSO.M5CQuerySequenceOnlyCrossHyenaRegressorModelB(
            seq_len=seq_len, hidden_dim=hidden, use_positional_encoding=pe,
            num_blocks=nblocks, fusion_type=fusion, use_atac=False,
        ).to(device)
    raise RuntimeError(
        "Could not infer the architecture from the checkpoint state_dict keys. "
        f"Sample keys: {sorted(keys)[:8]}"
    )


def cosine_lr_from_scheduler(sched_state: dict) -> float:
    """Best-effort LR for a CosineAnnealingLR state_dict (else NaN)."""
    try:
        base = float(sched_state["base_lrs"][0])
        eta_min = float(sched_state.get("eta_min", 0.0))
        t_max = float(sched_state["T_max"])
        last = float(sched_state["last_epoch"])
        return eta_min + (base - eta_min) * (1.0 + math.cos(math.pi * last / t_max)) / 2.0
    except Exception:
        return float("nan")


def rebuild_input_group_files(df_dmr, args):
    records = []
    if "input_group" in df_dmr.columns:
        cols = [c for c in ["m5c_bedgraph_path", "hm5c_bedgraph_path", "atac_bw_path", "rna_bw_path"] if c in df_dmr.columns]
        grouped = df_dmr.loc[:, ["input_group", *cols]].drop_duplicates().sort_values("input_group")
        for input_group, rows in grouped.groupby("input_group", sort=True):
            rec = {"input_group": int(input_group)}
            if "m5c_bedgraph_path" in rows.columns:
                rec["m5c_bedgraph"] = str(rows["m5c_bedgraph_path"].iloc[0])
            if "hm5c_bedgraph_path" in rows.columns:
                rec["hm5c_bedgraph"] = str(rows["hm5c_bedgraph_path"].iloc[0])
            if "atac_bw_path" in rows.columns:
                rec["atac_bw"] = str(rows["atac_bw_path"].iloc[0])
            if "rna_bw_path" in rows.columns:
                rec["rna_bw"] = str(rows["rna_bw_path"].iloc[0])
            records.append(rec)
    else:
        records.append({
            "input_group": 0,
            "m5c_bedgraph": args.m5c_bedgraph[0] if getattr(args, "m5c_bedgraph", None) else None,
            "hm5c_bedgraph": args.hm5c_bedgraph[0] if getattr(args, "hm5c_bedgraph", None) else None,
            "atac_bw": args.atac_bw[0] if getattr(args, "atac_bw", None) else None,
            "rna_bw": args.rna_coverage_bw[0] if getattr(args, "rna_coverage_bw", None) else None,
        })
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-script", default="run_m5c_query_crosshyena_dropmod.py",
                   help="The training script the checkpoint came from.")
    p.add_argument("--checkpoint", required=True, help="Path to the BEST checkpoint (.pt).")
    p.add_argument("--last-checkpoint", default=None,
                   help="Optional last checkpoint, used only to recover final_lr exactly.")
    p.add_argument("--out-csv", default=None, help="Default: the checkpoint's args.output_csv.")
    p.add_argument("--out-json", default=None, help="Default: the checkpoint's args.output_json.")
    p.add_argument("--device", type=int, default=None, help="GPU index (default: freest GPU).")
    p.add_argument("--dry-run", action="store_true", help="Evaluate and print, but write nothing.")
    return p.parse_args()


def main():
    cli = parse_args()

    print("=" * 70)
    print("[recover_results_from_checkpoint]")
    module = load_run_module(cli.run_script)
    prepare_fn = find_prepare_fn(module)
    print(f"  run script      : {cli.run_script}")
    print(f"  prepare fn      : {prepare_fn.__name__}")

    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args")
    if not ckpt_args:
        raise RuntimeError("checkpoint has no 'args' — cannot rebuild the data pipeline.")
    args = argparse.Namespace(**ckpt_args)
    ckpt_epoch = int(ckpt.get("epoch", -1))
    num_dmrs = int(ckpt.get("num_dmrs", getattr(args, "sample_sizes", [0])[0]))
    metrics = ckpt.get("metrics", {}) or {}
    state_dict = ckpt["model_state_dict"]
    print(f"  checkpoint epoch: {ckpt_epoch}  | num_dmrs={num_dmrs}  | metrics={metrics}")

    if torch.cuda.is_available():
        idx = cli.device if cli.device is not None else get_freest_gpu()
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    print(f"  device          : {device}")

    # ---- rebuild data exactly as the training run did -------------------
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = module.load_data(
        args, lazy=getattr(args, "lazy", False)
    )
    if getattr(args, "chromosome", None) is not None and hasattr(module, "subset_dataset_by_chromosome"):
        df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = module.subset_dataset_by_chromosome(
            df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, args.chromosome
        )[1:]

    prepared = prepare_fn(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)
    print(f"  split           : train={prepared.train_regions} val={prepared.val_regions} "
          f"test={prepared.test_regions} groups={prepared.non_overlap_groups}")

    # ---- rebuild model + load weights -----------------------------------
    model = build_model(module, args, prepared, state_dict, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    print(f"  model           : {type(model).__name__}  (state_dict loaded, strict OK)")

    # ---- recompute final-val and held-out test --------------------------
    final_val_loss, final_val_r2, final_val_pearsonr = module.evaluate(model, prepared.val_loader, device)
    if prepared.test_loader is not None:
        test_loss, test_r2, test_pearsonr = module.evaluate(model, prepared.test_loader, device)
    else:
        test_loss = test_r2 = test_pearsonr = float("nan")
    print(f"  recomputed final_val : loss={final_val_loss:.4f} R²={final_val_r2:.4f} r={final_val_pearsonr:.4f}")
    print(f"  recomputed test      : loss={test_loss:.4f} R²={test_r2:.4f} r={test_pearsonr:.4f}")

    # ---- final_lr (best-effort) -----------------------------------------
    final_lr = float("nan")
    sched_src = cli.last_checkpoint or cli.checkpoint
    try:
        sched = torch.load(sched_src, map_location="cpu", weights_only=False).get("scheduler_state_dict")
        if sched:
            final_lr = cosine_lr_from_scheduler(sched)
    except Exception:
        pass

    # ---- assemble a result row matching <module>.ExperimentResult -------
    ts = getattr(args, "timestamp", "")
    signal_h5ad = args.prediction_signal_h5ad.format(sample_size=prepared.usable_dmrs, timestamp=ts) \
        if getattr(args, "prediction_signal_h5ad", None) else ""
    regression_plot = args.regression_plot_path.format(sample_size=prepared.usable_dmrs, timestamp=ts) \
        if getattr(args, "regression_plot_path", None) else ""
    best_ckpt_path = args.best_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=ts) \
        if getattr(args, "best_checkpoint_path", None) else cli.checkpoint
    last_ckpt_path = args.last_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=ts) \
        if getattr(args, "last_checkpoint_path", None) else ""

    values = {
        "num_dmrs": prepared.usable_dmrs,
        "chromosome": getattr(args, "chromosome", None),
        "query_modality": getattr(args, "query_modality", None),
        "context_modality": getattr(args, "context_modality", None),
        "input_group_files": rebuild_input_group_files(df_dmr, args),
        "output_files": {
            "results_csv": cli.out_csv or args.output_csv,
            "results_json": cli.out_json or args.output_json,
            "signal_h5ad": signal_h5ad,
            "regression_plot": regression_plot,
            "checkpoints": {"best": best_ckpt_path, "last": last_ckpt_path},
        },
        "train_regions": prepared.train_regions,
        "val_regions": prepared.val_regions,
        "test_regions": prepared.test_regions,
        "non_overlap_groups": prepared.non_overlap_groups,
        "best_epoch": ckpt_epoch,
        "final_lr": final_lr,
        "best_val_loss": float(metrics.get("val_loss", float("nan"))),
        "best_val_r2": float(metrics.get("val_r2", float("nan"))),
        "best_val_pearsonr": float(metrics.get("val_pearsonr", float("nan"))),
        "final_val_loss": final_val_loss,
        "final_val_r2": final_val_r2,
        "final_val_pearsonr": final_val_pearsonr,
        "test_loss": test_loss,
        "test_r2": test_r2,
        "test_pearsonr": test_pearsonr,
        "split_info": prepared.split_info,
        "signal_h5ad": signal_h5ad,
        "regression_plot": regression_plot,
        "checkpoint_paths": {"best": best_ckpt_path, "last": last_ckpt_path},
    }

    # keep the field set/order of the target script's ExperimentResult
    import dataclasses
    field_names = [f.name for f in dataclasses.fields(module.ExperimentResult)]
    result = {}
    for name in field_names:
        if name in values:
            result[name] = values[name]
        else:
            print(f"  [warn] ExperimentResult field '{name}' not reconstructible; set to None")
            result[name] = None

    out_csv = cli.out_csv or args.output_csv
    out_json = cli.out_json or args.output_json
    print(f"  -> CSV : {out_csv}")
    print(f"  -> JSON: {out_json}")

    if cli.dry_run:
        print("[dry-run] nothing written.")
        print(json.dumps(result, indent=2, default=str))
        return

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "args": vars(args),
                "results": [result],
                "_recovery": {
                    "recovered_from_checkpoint": str(Path(cli.checkpoint).resolve()),
                    "run_script": str(Path(cli.run_script).resolve()),
                    "recovered_at": datetime.now().isoformat(timespec="seconds"),
                    "note": "test_* / final_val_* recomputed from the checkpoint weights; "
                            "best_* read from the checkpoint metrics.",
                },
            },
            fh,
            indent=2,
            default=str,
        )
    print("done.")


if __name__ == "__main__":
    main()
