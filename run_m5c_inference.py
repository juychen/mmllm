#!/usr/bin/env python
"""
Load a trained M5C-Query-Sequence-ATAC checkpoint and run inference only (no training).

Usage:
  python run_m5c_inference.py --checkpoint <path_to_best.pt> [--output-dir <dir>]

The script reads model configuration from the checkpoint's `args` dict,
rebuilds the model, loads weights, and runs inference on the validation
(or all) data, exporting predictions and metrics.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import pyfaidx

from data import (
    LazyM5cSequenceAtacDataset,
    add_clip_at_zero_argument,
    assign_non_overlapping_groups,
    ensure_path_list,
    get_sequence,
    load_data,
)
from models import (
    M5CQuerySequenceAtacCrossHyenaRegressor,
    M5CQuerySequenceAtacCrossHyenaRegressorModelB,
)
from utils import (
    export_prediction_signals_h5ad,
    get_freest_gpu,
    plot_regression_predictions,
    resolve_sample_sizes,
    set_random_seed,
)


# ---------------------------------------------------------------------------
# pure inference helpers (copied / adapted from the training script)
# ---------------------------------------------------------------------------

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (pred - target).pow(2)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    targets = []
    masks = []
    with torch.no_grad():
        for m5c_batch, sequence_batch, atac_batch, target_batch, mask_batch in loader:
            m5c_batch = m5c_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            pred = model(m5c_batch, sequence_batch, atac_batch)
            loss = masked_mse_loss(pred, target_batch, mask_batch)

            batch_count = m5c_batch.size(0)
            total_loss += loss.item() * batch_count
            total_count += batch_count
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)

    ss_res = (((targets - preds) ** 2) * masks).sum()
    masked_targets = targets[masks.bool()]
    masked_preds = preds[masks.bool()]
    target_mean = masked_targets.mean() if masked_targets.numel() > 0 else torch.tensor(0.0)
    ss_tot = (((targets - target_mean) ** 2) * masks).sum().clamp_min(1e-12)
    r2 = 1.0 - (ss_res / ss_tot)

    if masked_targets.numel() > 1:
        centered_targets = masked_targets - masked_targets.mean()
        centered_preds = masked_preds - masked_preds.mean()
        pearson_denom = centered_targets.pow(2).sum().sqrt() * centered_preds.pow(2).sum().sqrt()
        pearson_r = (centered_targets * centered_preds).sum() / pearson_denom.clamp_min(1e-12)
        pearson_r_value = pearson_r.item()
    else:
        pearson_r_value = float("nan")

    return total_loss / max(total_count, 1), r2.item(), pearson_r_value


def collect_predictions(model: nn.Module, loader, device: torch.device):
    model.eval()
    preds = []
    targets = []
    masks = []
    with torch.no_grad():
        for m5c_batch, sequence_batch, atac_batch, target_batch, mask_batch in loader:
            m5c_batch = m5c_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            pred = model(m5c_batch, sequence_batch, atac_batch)
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), torch.cat(masks, dim=0)


def normalize_chromosome_label(chromosome: str) -> str:
    chrom_value = str(chromosome).strip()
    if not chrom_value:
        raise ValueError("--chromosome cannot be empty.")
    return chrom_value if chrom_value.lower().startswith("chr") else f"chr{chrom_value}"


def subset_dataset_by_chromosome(df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, chromosome: str):
    chrom_series = df_dmr["chr"].astype(str).map(normalize_chromosome_label)
    matched_indices = np.flatnonzero(chrom_series == chromosome)
    if matched_indices.size == 0:
        available = sorted(chrom_series.drop_duplicates().tolist())
        raise ValueError(
            f"No DMR rows for chromosome {chromosome}. Available: {', '.join(available)}"
        )
    filtered_df = df_dmr.iloc[matched_indices].reset_index(drop=True)
    idx_list = matched_indices.tolist()
    return (
        filtered_df,
        [seqs[i] for i in idx_list],
        [mcg_tracks[i] for i in idx_list],
        [hmcg_tracks[i] for i in idx_list],
        [atac_tracks[i] for i in idx_list],
    )


# ---------------------------------------------------------------------------
# data preparation  (single loader for inference — no train/val split needed)
# ---------------------------------------------------------------------------

@dataclass
class PreparedInferenceData:
    loader: torch.utils.data.DataLoader
    num_regions: int
    seq_len: int
    region_metadata: pd.DataFrame


def prepare_inference_data(args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, sample_size=None):
    """Build a single Dataset / DataLoader covering all regions for inference."""
    usable = min(len(df_dmr), len(seqs), len(mcg_tracks), len(hmcg_tracks), len(atac_tracks))
    if sample_size is not None:
        usable = min(usable, sample_size)
    seq_len = args.target_length

    # fetch real sequences for metadata  (open / close before forking workers)
    genome = pyfaidx.Fasta(args.genome_fasta)

    split_regions_df = (
        df_dmr.iloc[:usable]
        .copy()
        .reset_index()
        .rename(columns={"index": "original_idx"})
    )
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)

    all_indices = list(range(usable))
    all_subset = split_regions_df.iloc[all_indices].copy().reset_index(drop=True)
    all_seqs = []
    for _, row in all_subset.iterrows():
        chrom_name = str(row["chr"]).removeprefix("chr")
        chrom = "chr" + chrom_name
        s = int(row["start_expanded"])
        e = int(row["end_expanded"])
        all_seqs.append(get_sequence(chrom, s, e, genome))
    all_subset["sequence"] = [str(s)[:seq_len].upper() for s in all_seqs]

    try:
        genome.close()
    except Exception:
        pass

    hm5c_paths = ensure_path_list(getattr(args, "hm5c_bedgraph", None))
    m5c_paths = ensure_path_list(getattr(args, "m5c_bedgraph", None))
    atac_paths = ensure_path_list(getattr(args, "atac_bw", None))

    dataset = LazyM5cSequenceAtacDataset(
        indices=all_indices,
        df_dmr=split_regions_df,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=m5c_paths[0],
        hm5c_bedgraph=hm5c_paths[0],
        atac_bw_path=atac_paths[0],
        target_length=args.target_length,
        mask_mode=args.mask_mode,
        atac_scaling=args.atac_scaling,
        augment_rc=False,  # no augmentation during inference
        clip_at_zero=args.clip_at_zero,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True,
    )

    return PreparedInferenceData(
        loader=loader,
        num_regions=len(dataset),
        seq_len=seq_len,
        region_metadata=all_subset,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _guess_experiment_json(checkpoint_path: Path) -> Path | None:
    """Attempt to find the experiment JSON from the checkpoint filename.

    Checkpoint pattern:  {timestamp}_{run_label}_best_{sample_size}.pt
                           or {timestamp}_{run_label}_last_{sample_size}.pt
    JSON pattern:        {timestamp}_{run_label}_results.json

    Falls back to trying all *.json in the same directory if pattern doesn't match.
    """
    name = checkpoint_path.stem  # strip .pt
    # Remove _best_NNN or _last_NNN suffix
    import re
    base = re.sub(r"_(best|last)_\d+$", "", name)
    direct = checkpoint_path.with_name(f"{base}_results.json")
    if direct.is_file():
        return direct

    # Fallback: look for any results.json with matching timestamp prefix
    for f in sorted(checkpoint_path.parent.glob("*_results.json")):
        if f.stem.startswith(base[:20]):  # timestamp prefix ~20 chars
            return f

    # Last resort: any .json
    jsons = sorted(checkpoint_path.parent.glob("*.json"))
    return jsons[0] if jsons else None


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Load a trained checkpoint and run inference."
    )
    # required
    p.add_argument("--checkpoint", required=True,
                   help="Path to a trained .pt checkpoint (best or last).")
    p.add_argument("--experiment-json", default=None,
                   help="Path to the experiment results.json. Auto-detected from checkpoint dir if omitted.")

    # optional overrides (defaults match training script; usually read from ckpt)
    p.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv",
                   help="Path to DMR file.")
    p.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    p.add_argument("--m5c-bedgraph", nargs="+",
                   default=["/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz"])
    p.add_argument("--hm5c-bedgraph", nargs="+",
                   default=["/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"])
    p.add_argument("--atac-bw", nargs="+",
                   default=["/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw"])
    p.add_argument("--chromosome", default=None)
    p.add_argument("--target-length", type=int, default=1024)
    p.add_argument("--sample-sizes", nargs="+", type=str, default=["all"],
                   help="Number of regions to use for inference. Use 'all' for all regions, "
                        "or specify integers like 1000 2000. Default: all.")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--mask-mode", choices=["cpg_both", "cpg_forward", "c_only", "all"], default="cpg_both")
    p.add_argument("--atac-scaling", choices=["none", "minmax"], default="minmax")
    add_clip_at_zero_argument(p)
    p.add_argument("--use-positional-encoding", action="store_true")
    p.add_argument("--model-name", choices=["baseline", "model_b"], default="baseline")
    p.add_argument("--model-b-blocks", type=int, default=2)
    p.add_argument("--model-b-fusion", choices=["cross_hyena", "cross_attention"], default="cross_hyena")

    # output
    p.add_argument("--output-dir", default="output/inference")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--use-all-input-groups", action="store_true")
    p.add_argument("--lazy", action="store_true",
                   help="Enable lazy loading: fetch data on-the-fly per batch.")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --- load experiment JSON for model / training config ---
    exp_json_path = None
    if args.experiment_json:
        exp_json_path = Path(args.experiment_json)
    else:
        exp_json_path = _guess_experiment_json(ckpt_path)

    json_args_used = False
    if exp_json_path and exp_json_path.is_file():
        with open(exp_json_path) as fj:
            exp_data = json.load(fj)
        exp_args = exp_data.get("args", {})
        print(f"Reading config from: {exp_json_path.name}")
        # Apply JSON args for keys that user didn't override from CLI
        for key, val in exp_args.items():
            if key in ("output_csv", "output_json", "prediction_signal_csv",
                       "regression_plot_path", "best_checkpoint_path",
                       "last_checkpoint_path", "sample_sizes", "timestamp"):
                continue  # output paths / sample sizes irrelevant for inference
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, val)
        json_args_used = True
    else:
        print(f"No experiment JSON found (searched near {ckpt_path}). "
              f"Using CLI defaults / checkpoint args as fallback.")

    # --- load checkpoint ---
    device = torch.device(f"cuda:{get_freest_gpu()}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path.name}")
    print(f"  epoch: {checkpoint.get('epoch', '?')}")
    if "metrics" in checkpoint:
        print(f"  val_loss: {checkpoint['metrics'].get('val_loss', '?'):.6f}")
        print(f"  val_r2:   {checkpoint['metrics'].get('val_r2', '?'):.6f}")

    # Merge checkpoint args into CLI args (CLI overrides ckpt, ckpt overrides defaults)
    ckpt_args = checkpoint.get("args", {})
    for key, val in ckpt_args.items():
        if key in ("output_csv", "output_json", "prediction_signal_csv",
                   "regression_plot_path", "best_checkpoint_path",
                   "last_checkpoint_path", "sample_sizes", "timestamp"):
            continue
        if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
            setattr(args, key, val)

    set_random_seed(args.seed)

    # --- resolve sample sizes ---
    sample_sizes = resolve_sample_sizes(args.sample_sizes, args)
    sample_size = sample_sizes[0]  # use the first sample size for inference
    print(f"Using sample size: {sample_size if sample_size != float('inf') else 'all'}")

    # IMPORTANT: update args so load_data sees integer sample sizes (avoids TypeError in max())
    args.sample_sizes = sample_sizes

    # --- load data ---
    args.use_m5c = True
    if args.chromosome is not None:
        args.chromosome = normalize_chromosome_label(args.chromosome)

    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(
        args, lazy=getattr(args, "lazy", False),
    )

    if args.chromosome is not None:
        df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = subset_dataset_by_chromosome(
            df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, args.chromosome,
        )

    prepared = prepare_inference_data(args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, sample_size=sample_size)

    # --- build model ---
    if args.model_name == "model_b":
        model = M5CQuerySequenceAtacCrossHyenaRegressorModelB(
            seq_len=prepared.seq_len,
            hidden_dim=args.hidden_dim,
            use_positional_encoding=args.use_positional_encoding,
            num_blocks=args.model_b_blocks,
            fusion_type=args.model_b_fusion,
        ).to(device)
    else:
        model = M5CQuerySequenceAtacCrossHyenaRegressor(
            seq_len=prepared.seq_len,
            hidden_dim=args.hidden_dim,
            post_filter_len=min(prepared.seq_len, 4),
            use_positional_encoding=args.use_positional_encoding,
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded: {args.model_name}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    # --- inference ---
    print(f"\nRunning inference on {prepared.num_regions:,} regions ...")
    val_loss, val_r2, val_pearsonr = evaluate(model, prepared.loader, device)
    all_preds, all_targets, all_masks = collect_predictions(model, prepared.loader, device)

    print(f"  Loss:     {val_loss:.6f}")
    print(f"  R²:       {val_r2:.6f}")
    print(f"  Pearson r:{val_pearsonr:.6f}")

    # --- export ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = ckpt_path.stem  # e.g. "2025-01-01-12-00-00_...best_500000"

    signal_h5ad = out_dir / f"{ckpt_stem}_inference_predictions.h5ad"
    regression_plot = out_dir / f"{ckpt_stem}_inference_regression.png"

    export_prediction_signals_h5ad(
        str(signal_h5ad),
        prepared.region_metadata,
        all_preds.squeeze(-1).numpy(),
        all_targets.squeeze(-1).numpy(),
        all_masks.squeeze(-1).numpy(),
    )
    print(f"Predictions saved to: {signal_h5ad}")

    plot_regression_predictions(
        str(regression_plot),
        all_preds.squeeze(-1).numpy(),
        all_targets.squeeze(-1).numpy(),
        all_masks.squeeze(-1).numpy(),
        title=f"5mC-query seq+ATAC → 5hmC  ({ckpt_stem})\n"
              f"R²={val_r2:.4f}  Pearson r={val_pearsonr:.4f}  n={prepared.num_regions:,}",
    )
    print(f"Plot saved to: {regression_plot}")

    # --- summary json ---
    metrics_json = out_dir / f"{ckpt_stem}_inference_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "num_regions": prepared.num_regions,
                "seq_len": prepared.seq_len,
                "loss": val_loss,
                "r2": val_r2,
                "pearson_r": val_pearsonr,
            },
            f,
            indent=2,
        )
    print(f"Metrics saved to: {metrics_json}")


if __name__ == "__main__":
    main()
