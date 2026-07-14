"""Trainer that supports 5mC → 5hmC prediction with ATAC **optional**.

Near-clone of `run_m5c_query_sequence_atac_crosshyena_experiments.py`
with these differences:

- `--atac-bw` defaults to None (omitted).
- Dataset returns zero-filled ATAC when no bigWig is provided.
- Model uses `M5CQuerySequenceOnlyCrossHyenaRegressorModelB`, which
  internally skips the ATAC projection when ATAC is absent — no
  zero-vector hack.
- All optimizations preserved: AMP, gradient checkpointing, gradient
  accumulation, lazy loading, RC augmentation, cosine scheduler, etc.

Why a separate file?
- Keeps the main branch semantically "ATAC is required" intact.
- This script is for ablation studies (e.g., "5mC + sequence only").
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import pyfaidx

from data import (
    assign_non_overlapping_groups,
    ensure_path_list,
    get_sequence,
    load_data,
)
from data_sequence_only import LazyM5cSequenceOnlyDataset
from models_sequence_only import M5CQuerySequenceOnlyCrossHyenaRegressorModelB
from utils import (
    export_prediction_signals,
    get_freest_gpu,
    plot_regression_predictions,
    resolve_sample_sizes,
    set_random_seed,
)


def masked_mse_loss(pred, target, mask):
    squared_error = (pred - target).pow(2)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def build_scheduler(optimizer, args, steps_per_experiment):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        total_steps = args.scheduler_t_max if args.scheduler_t_max > 0 else steps_per_experiment
        total_steps = max(1, total_steps)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.scheduler_min_lr
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def build_optimizer(model, args):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, metrics, args, num_dmrs):
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    torch.save(
        {
            "epoch": epoch,
            "num_dmrs": num_dmrs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "metrics": metrics,
            "args": vars(args),
        },
        checkpoint_file,
    )


@dataclass
class PreparedSequenceData:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    usable_dmrs: int
    seq_len: int
    train_regions: int
    val_regions: int
    non_overlap_groups: int
    val_region_metadata: pd.DataFrame


def prepare_sequence_data(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks):
    if not mcg_tracks:
        raise ValueError("5mC tracks required for 5mC→5hmC prediction. Set --m5c-bedgraph to a valid path.")

    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(mcg_tracks), len(hmcg_tracks))
    seq_len = args.target_length

    genome = pyfaidx.Fasta(args.genome_fasta)

    split_regions_df = df_dmr.iloc[:usable_dmrs].copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
    split_regions_df = assign_non_overlapping_groups(
        split_regions_df, "chr", "start_expanded", "end_expanded"
    )

    group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * args.train_ratio))
    train_group_ids = set(group_ids[:num_train_groups].tolist())
    train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
    train_indices = np.flatnonzero(train_mask).tolist()
    val_indices = np.flatnonzero(~train_mask).tolist()

    val_subset = split_regions_df.iloc[val_indices].copy().reset_index(drop=True)
    val_seqs = []
    for _, row in val_subset.iterrows():
        chrom_name = str(row["chr"]).removeprefix("chr")
        chrom = "chr" + chrom_name
        s = int(row["start_expanded"])
        e = int(row["end_expanded"])
        val_seqs.append(get_sequence(chrom, s, e, genome))
    val_subset["sequence"] = [str(s)[:seq_len].upper() for s in val_seqs]
    val_region_metadata = val_subset

    try:
        genome.close()
    except Exception:
        pass

    hm5c_paths = ensure_path_list(getattr(args, "hm5c_bedgraph", None))
    m5c_paths = ensure_path_list(getattr(args, "m5c_bedgraph", None))
    atac_paths = ensure_path_list(getattr(args, "atac_bw", None))
    atac_path = atac_paths[0] if atac_paths else None

    train_dataset = LazyM5cSequenceOnlyDataset(
        indices=train_indices,
        df_dmr=split_regions_df,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=m5c_paths[0],
        hm5c_bedgraph=hm5c_paths[0],
        atac_bw_path=atac_path,
        target_length=args.target_length,
        mask_mode=args.mask_mode,
        atac_scaling=args.atac_scaling,
        augment_rc=getattr(args, "augment_reverse_complement", False),
    )
    val_dataset = LazyM5cSequenceOnlyDataset(
        indices=val_indices,
        df_dmr=split_regions_df,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=m5c_paths[0],
        hm5c_bedgraph=hm5c_paths[0],
        atac_bw_path=atac_path,
        target_length=args.target_length,
        mask_mode=args.mask_mode,
        atac_scaling=args.atac_scaling,
        augment_rc=False,
    )

    return PreparedSequenceData(
        train_loader=torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=2, prefetch_factor=2, persistent_workers=False,
            pin_memory=True,
        ),
        val_loader=torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=2, prefetch_factor=2, persistent_workers=False,
            pin_memory=True,
        ),
        usable_dmrs=usable_dmrs,
        seq_len=seq_len,
        train_regions=len(train_dataset),
        val_regions=len(val_dataset),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
        val_region_metadata=val_region_metadata,
    )


def normalize_chromosome_label(chromosome):
    chrom_value = str(chromosome).strip()
    if not chrom_value:
        raise ValueError("--chromosome cannot be empty.")
    return chrom_value if chrom_value.lower().startswith("chr") else f"chr{chrom_value}"


def subset_dataset_by_chromosome(df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, chromosome):
    normalized_chromosome = normalize_chromosome_label(chromosome)
    chrom_series = df_dmr["chr"].astype(str).map(normalize_chromosome_label)
    matched_indices = np.flatnonzero(chrom_series == normalized_chromosome)
    if matched_indices.size == 0:
        available = sorted(chrom_series.drop_duplicates().tolist())
        raise ValueError(
            f"No DMR rows found for chromosome {normalized_chromosome}. "
            f"Available: {', '.join(available)}"
        )
    filtered_df = df_dmr.iloc[matched_indices].reset_index(drop=True)
    index_list = matched_indices.tolist()
    return (
        normalized_chromosome,
        filtered_df,
        [seqs[idx] for idx in index_list],
        [mcg_tracks[idx] for idx in index_list],
        [hmcg_tracks[idx] for idx in index_list],
        [atac_tracks[idx] for idx in index_list] if atac_tracks else [],
    )


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for m5c_batch, sequence_batch, atac_batch, target_batch, mask_batch, _ in loader:
            m5c_batch = m5c_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device) if model.use_atac else None
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
        centered_t = masked_targets - masked_targets.mean()
        centered_p = masked_preds - masked_preds.mean()
        denom = centered_t.pow(2).sum().sqrt() * centered_p.pow(2).sum().sqrt()
        pearson_r = (centered_t * centered_p).sum() / denom.clamp_min(1e-12)
        pearson_r_value = pearson_r.item()
    else:
        pearson_r_value = float("nan")
    return total_loss / max(total_count, 1), r2.item(), pearson_r_value


def collect_predictions(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for m5c_batch, sequence_batch, atac_batch, target_batch, mask_batch, _ in loader:
            m5c_batch = m5c_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device) if model.use_atac else None
            pred = model(m5c_batch, sequence_batch, atac_batch)
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), torch.cat(masks, dim=0)


@dataclass
class ExperimentResult:
    num_dmrs: int
    chromosome: str | None
    input_group_files: list[dict]
    output_files: dict
    train_regions: int
    val_regions: int
    non_overlap_groups: int
    best_epoch: int
    final_lr: float
    best_val_loss: float
    best_val_r2: float
    best_val_pearsonr: float
    final_val_loss: float
    final_val_r2: float
    final_val_pearsonr: float
    signal_csv: str
    regression_plot: str
    checkpoint_paths: dict


def run_experiment(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks):
    device = torch.device(f"cuda:{get_freest_gpu()}" if torch.cuda.is_available() else "cpu")
    prepared = prepare_sequence_data(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)

    model = M5CQuerySequenceOnlyCrossHyenaRegressorModelB(
        seq_len=prepared.seq_len,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=args.use_positional_encoding,
        num_blocks=args.model_b_blocks,
        fusion_type=args.model_b_fusion,
        use_atac=args.use_atac,
    ).to(device)

    # Gradient checkpointing: override forward to call torch.checkpoint on each block
    if args.gradient_checkpointing:
        try:
            import torch.utils.checkpoint as ckpt
            def checkpointed_forward(m5c_track, sequence_track, atac_track):
                x = model.query_norm(model.query_proj(m5c_track))
                seq_h = model.sequence_norm(model.sequence_proj(sequence_track))
                if model.use_atac and atac_track is not None:
                    atac_h = model.atac_norm(model.atac_proj(atac_track))
                    ctx_in = torch.cat([seq_h, atac_h], dim=-1)
                else:
                    ctx_in = seq_h
                ctx = model.context_norm(model.context_proj(ctx_in))
                if model.position_encoding is not None:
                    x = model.position_encoding(x)
                    ctx = model.position_encoding(ctx)
                for blk in model.blocks:
                    x = ckpt.checkpoint(blk, x, ctx, use_reentrant=False)
                return model.head(model.final_norm(x))
            model.forward = checkpointed_forward
        except (ImportError, AttributeError) as e:
            print(f"Warning: gradient checkpointing not available ({e}), falling back.")
            args.gradient_checkpointing = False

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.num_epochs)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    amp_dtype = torch.bfloat16 if args.amp else torch.float32

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_r2 = float("nan")
    best_val_pearsonr = float("nan")
    best_state = None
    last_epoch = 0
    best_checkpoint_path = args.best_checkpoint_path.format(
        sample_size=prepared.usable_dmrs, timestamp=args.timestamp
    )
    last_checkpoint_path = args.last_checkpoint_path.format(
        sample_size=prepared.usable_dmrs, timestamp=args.timestamp
    )
    patience_left = args.patience

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        model.train()
        running_loss, seen, accum_count = 0.0, 0, 0
        optimizer.zero_grad()

        for m5c_batch, sequence_batch, atac_batch, target_batch, mask_batch, _ in prepared.train_loader:
            m5c_batch = m5c_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device) if model.use_atac else None
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                pred = model(m5c_batch, sequence_batch, atac_batch)
                loss = masked_mse_loss(pred, target_batch, mask_batch)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accum_count += 1

            batch_count = m5c_batch.size(0)
            running_loss += loss.item() * batch_count * args.gradient_accumulation_steps
            seen += batch_count

            if accum_count % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss = running_loss / max(seen, 1)
        val_loss, val_r2, val_pearsonr = evaluate(model, prepared.val_loader, device)
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[num_dmrs={prepared.usable_dmrs}] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_r2={val_r2:.4f} | val_pearsonr={val_pearsonr:.4f} | lr={current_lr:.6g}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_val_pearsonr = val_pearsonr
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(
                best_checkpoint_path, model, optimizer, scheduler, epoch,
                {"train_loss": train_loss, "val_loss": val_loss, "val_r2": val_r2,
                 "val_pearsonr": val_pearsonr, "is_best": True},
                args, prepared.usable_dmrs,
            )
            patience_left = args.patience
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                break

    save_checkpoint(
        last_checkpoint_path, model, optimizer, scheduler, last_epoch,
        {"val_loss": val_loss, "val_r2": val_r2, "val_pearsonr": val_pearsonr,
         "is_best": last_epoch == best_epoch},
        args, prepared.usable_dmrs,
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_r2, final_val_pearsonr = evaluate(model, prepared.val_loader, device)
    final_preds, final_targets, final_masks = collect_predictions(model, prepared.val_loader, device)

    signal_csv = args.prediction_signal_csv.format(
        sample_size=prepared.usable_dmrs, timestamp=args.timestamp
    )
    regression_plot = args.regression_plot_path.format(
        sample_size=prepared.usable_dmrs, timestamp=args.timestamp
    )
    export_prediction_signals(
        signal_csv,
        prepared.val_region_metadata,
        final_preds.squeeze(-1).numpy(),
        final_targets.squeeze(-1).numpy(),
        final_masks.squeeze(-1).numpy(),
    )
    plot_regression_predictions(
        regression_plot,
        final_preds.squeeze(-1).numpy(),
        final_targets.squeeze(-1).numpy(),
        final_masks.squeeze(-1).numpy(),
        title=f"5mC-query {'sequence+ATAC' if args.use_atac else 'sequence-only'} -> 5hmC (n={prepared.usable_dmrs})",
    )

    input_group_files = [{
        "input_group": 0,
        "m5c_bedgraph": args.m5c_bedgraph[0] if getattr(args, "m5c_bedgraph", None) else None,
        "hm5c_bedgraph": args.hm5c_bedgraph[0] if getattr(args, "hm5c_bedgraph", None) else None,
        "atac_bw": args.atac_bw[0] if getattr(args, "atac_bw", None) else None,
        "use_atac": args.use_atac,
    }]

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        chromosome=getattr(args, "chromosome", None),
        input_group_files=input_group_files,
        output_files={
            "results_csv": args.output_csv,
            "results_json": args.output_json,
            "signal_csv": signal_csv,
            "regression_plot": regression_plot,
            "checkpoints": {"best": best_checkpoint_path, "last": last_checkpoint_path},
        },
        train_regions=prepared.train_regions,
        val_regions=prepared.val_regions,
        non_overlap_groups=prepared.non_overlap_groups,
        best_epoch=best_epoch,
        final_lr=optimizer.param_groups[0]["lr"],
        best_val_loss=best_val_loss,
        best_val_r2=best_val_r2,
        best_val_pearsonr=best_val_pearsonr,
        final_val_loss=final_val_loss,
        final_val_r2=final_val_r2,
        final_val_pearsonr=final_val_pearsonr,
        signal_csv=signal_csv,
        regression_plot=regression_plot,
        checkpoint_paths={"best": best_checkpoint_path, "last": last_checkpoint_path},
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 5mC-query sequence(+optional ATAC)-context -> 5hmC experiments. ATAC is optional via --atac-bw."
    )
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv",
                        help="Path to DMR file. Supports CSV (chr/start/end/length/center columns) or BED (chr, start, end as first three columns).")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument(
        "--m5c-bedgraph", nargs="+",
        default=["/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz"],
    )
    parser.add_argument(
        "--hm5c-bedgraph", nargs="+",
        default=["/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"],
    )
    parser.add_argument(
        "--atac-bw", nargs="+", default=None,
        help="Optional ATAC-seq bigWig. If omitted, model uses sequence-only context.",
    )
    parser.add_argument("--use-atac", action="store_true",
                        help="Enable ATAC branch in the model. Requires --atac-bw.")
    parser.add_argument("--sample-sizes", nargs="+", type=str, required=True)
    parser.add_argument("--chromosome", default=None)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--model-b-blocks", type=int, default=2)
    parser.add_argument("--model-b-fusion", choices=["cross_hyena", "cross_attention"], default="cross_hyena")
    parser.add_argument("--use-positional-encoding", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--atac-scaling", choices=["none", "minmax"], default="none")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (bfloat16). Recommended for long sequences.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Accumulate gradients over N mini-batches. Use with --batch-size 4-8 for long sequences.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade compute for memory. Recommended for very long sequences (8k+).")
    parser.add_argument("--mask-mode", choices=["cpg_both", "cpg_forward", "all"], default="cpg_forward")
    parser.add_argument("--augment-reverse-complement", action="store_true")
    parser.add_argument("--lazy", action="store_true",
                        help="Fetch sequence/track data on-the-fly per batch instead of preloading.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", default="output/m5c_query_sequence_only_results.csv")
    parser.add_argument("--output-json", default="output/m5c_query_sequence_only_results.json")
    parser.add_argument("--timestamp", default="")
    parser.add_argument(
        "--prediction-signal-csv",
        default="output/{timestamp}_m5c_query_sequence_only_prediction_signals_{sample_size}.csv",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="output/{timestamp}_m5c_query_sequence_only_regression_plot_{sample_size}.png",
    )
    parser.add_argument(
        "--best-checkpoint-path",
        default="output/{timestamp}_m5c_query_sequence_only_best_{sample_size}.pt",
    )
    parser.add_argument(
        "--last-checkpoint-path",
        default="output/{timestamp}_m5c_query_sequence_only_last_{sample_size}.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.sample_sizes = resolve_sample_sizes(args.sample_sizes, args)
    args.use_m5c = True
    if args.chromosome is not None:
        args.chromosome = normalize_chromosome_label(args.chromosome)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(
        args, lazy=getattr(args, "lazy", False)
    )
    if args.chromosome is not None:
        df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = subset_dataset_by_chromosome(
            df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, args.chromosome
        )[1:]

    results = []
    for sample_size in args.sample_sizes:
        results.append(asdict(
            run_experiment(sample_size, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)
        ))

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
