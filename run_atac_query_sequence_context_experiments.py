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
    add_clip_at_zero_argument,
    assign_non_overlapping_groups,
    ensure_path_list,
    get_sequence,
    load_data,
)
from data import LazyM5cSequenceAtacDataset  # reused: tuple (m5c, seq, atac, hm5c, mask)
from models import MinimalCrossHyenaRegressor
from utils import (
    export_prediction_signals_h5ad,
    get_freest_gpu,
    plot_regression_predictions,
    resolve_sample_sizes,
    set_random_seed,
)


@dataclass
class PreparedAtacSequenceData:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    usable_dmrs: int
    seq_len: int
    post_filter_len: int
    train_regions: int
    val_regions: int
    non_overlap_groups: int
    val_region_metadata: pd.DataFrame


@dataclass
class ExperimentResult:
    num_dmrs: int
    query_modality: str
    context_modality: str
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
    signal_h5ad: str
    regression_plot: str
    checkpoint_paths: dict


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (pred - target).pow(2)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def build_scheduler(optimizer: torch.optim.Optimizer, args, steps_per_experiment: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        total_steps = args.scheduler_t_max if args.scheduler_t_max > 0 else steps_per_experiment
        total_steps = max(1, total_steps)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.scheduler_min_lr,
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=args.learning_rate)


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


def prepare_atac_query_sequence_context_data(
    num_dmrs: int,
    args,
    df_dmr,
    seqs,
    _mcg_tracks,
    hmcg_tracks,
    atac_tracks,
) -> PreparedAtacSequenceData:
    """Prepare train/val loaders using lazy on-the-fly loading.

    Reuses `LazyM5cSequenceAtacDataset` from `data.py` — its tuple
    `(m5c, seq, atac, hm5c, mask)` is laid out so that the **second slot
    (sequence) is the context** and the **third slot (atac) is the query**.
    The m5c slot is unused (consumed but discarded) since the upstream
    task is ATAC → 5hmC, not 5mC → 5hmC.
    """
    if not getattr(args, "lazy", False):
        raise ValueError(
            "This rewritten trainer requires --lazy. The old in-memory path is "
            "removed because it does not scale to 16kb sequences."
        )

    if not hmcg_tracks:
        raise ValueError("5hmC tracks required. Set --hm5c-bedgraph to a valid path.")
    if not atac_tracks:
        raise ValueError("ATAC tracks required. Set --atac-bw to a valid path.")

    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(hmcg_tracks), len(atac_tracks))
    seq_len = args.target_length
    post_filter_len = min(seq_len, 4)

    # Open genome briefly to fetch val sequences for metadata (no bigWig/Tabix
    # reads here — those happen lazily inside the Dataset worker).
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
    m5c_paths = ensure_path_list(getattr(args, "m5c_bedgraph", None)) or ["/dev/null"]
    atac_paths = ensure_path_list(getattr(args, "atac_bw", None))

    train_dataset = LazyM5cSequenceAtacDataset(
        indices=train_indices,
        df_dmr=split_regions_df,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=m5c_paths[0],   # unused downstream but required by __init__
        hm5c_bedgraph=hm5c_paths[0],
        atac_bw_path=atac_paths[0],
        target_length=args.target_length,
        mask_mode=args.mask_mode,
        atac_scaling=args.atac_scaling,
        augment_rc=getattr(args, "augment_reverse_complement", False),
        clip_at_zero=getattr(args, "clip_at_zero", False),
    )
    val_dataset = LazyM5cSequenceAtacDataset(
        indices=val_indices,
        df_dmr=split_regions_df,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=m5c_paths[0],
        hm5c_bedgraph=hm5c_paths[0],
        atac_bw_path=atac_paths[0],
        target_length=args.target_length,
        mask_mode=args.mask_mode,
        atac_scaling=args.atac_scaling,
        augment_rc=False,
        clip_at_zero=getattr(args, "clip_at_zero", False),
    )

    return PreparedAtacSequenceData(
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
        post_filter_len=post_filter_len,
        train_regions=len(train_dataset),
        val_regions=len(val_dataset),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
        val_region_metadata=val_region_metadata,
    )


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    targets = []
    masks = []
    with torch.no_grad():
        # Lazy dataset returns 5 items: (m5c_unused, seq_context, atac_query, hm5c_target, mask)
        for _m5c_unused, seq_batch, atac_batch, target_batch, mask_batch in loader:
            atac_batch = atac_batch.to(device)
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            pred = model(atac_batch, seq_batch)
            loss = masked_mse_loss(pred, target_batch, mask_batch)

            batch_count = atac_batch.size(0)
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
        # Lazy dataset returns 5 items: (m5c_unused, seq_context, atac_query, hm5c_target, mask)
        for _m5c_unused, seq_batch, atac_batch, target_batch, mask_batch in loader:
            atac_batch = atac_batch.to(device)
            seq_batch = seq_batch.to(device)
            pred = model(atac_batch, seq_batch)
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), torch.cat(masks, dim=0)


def run_experiment(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> ExperimentResult:
    device = torch.device(f"cuda:{get_freest_gpu()}" if torch.cuda.is_available() else "cpu")
    prepared = prepare_atac_query_sequence_context_data(
        num_dmrs,
        args,
        df_dmr,
        seqs,
        mcg_tracks,
        hmcg_tracks,
        atac_tracks,
    )

    model = MinimalCrossHyenaRegressor(
        seq_len=prepared.seq_len,
        query_dim=1,
        context_dim=4,
        hidden_dim=args.hidden_dim,
        post_filter_len=prepared.post_filter_len,
        use_positional_encoding=args.use_positional_encoding,
    ).to(device)

    # Gradient checkpointing: trade compute for memory on long sequences.
    if args.gradient_checkpointing:
        try:
            import torch.utils.checkpoint as ckpt
            orig_cross = model.cross
            orig_post_hyena = model.post_hyena
            def checkpointed_forward(query_track, context_track):
                q = model.query_proj(query_track)
                c = model.context_proj(context_track)
                if model.position_encoding is not None:
                    q = model.position_encoding(q)
                    c = model.position_encoding(c)
                h = ckpt.checkpoint(orig_cross, q, c, use_reentrant=False)
                h = h + model.cross_to_post(h)
                h = ckpt.checkpoint(orig_post_hyena, h, use_reentrant=False)
                h = model.norm(h)
                return model.head(h)
            model.forward = checkpointed_forward
        except (ImportError, AttributeError) as e:
            print(f"Warning: gradient checkpointing not available ({e}), falling back.")
            args.gradient_checkpointing = False

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.num_epochs)

    # Mixed precision (bfloat16) scaler
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
        running_loss = 0.0
        seen = 0
        optimizer.zero_grad()
        accum_count = 0

        # Lazy dataset returns 5 items: (m5c_unused, seq_context, atac_query, hm5c_target, mask)
        for _m5c_unused, seq_batch, atac_batch, target_batch, mask_batch in prepared.train_loader:
            atac_batch = atac_batch.to(device)
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                pred = model(atac_batch, seq_batch)
                loss = masked_mse_loss(pred, target_batch, mask_batch)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accum_count += 1

            batch_count = atac_batch.size(0)
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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

    signal_h5ad = args.prediction_signal_h5ad.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    regression_plot = args.regression_plot_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    export_prediction_signals_h5ad(
        signal_h5ad,
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
        title=f"ATAC Query vs Sequence Context (n={prepared.usable_dmrs})",
    )

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        query_modality="atac",
        context_modality="sequence",
        chromosome=getattr(args, "chromosome", None),
        input_group_files=[{
            "input_group": 0,
            "m5c_bedgraph": args.m5c_bedgraph[0] if getattr(args, "m5c_bedgraph", None) else None,
            "hm5c_bedgraph": args.hm5c_bedgraph[0] if getattr(args, "hm5c_bedgraph", None) else None,
            "atac_bw": args.atac_bw[0] if getattr(args, "atac_bw", None) else None,
        }],
        output_files={
            "results_csv": args.output_csv,
            "results_json": args.output_json,
            "signal_h5ad": signal_h5ad,
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
        signal_h5ad=signal_h5ad,
        regression_plot=regression_plot,
        checkpoint_paths={"best": best_checkpoint_path, "last": last_checkpoint_path},
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run non-overlap ATAC-query and sequence-context experiments for predicting 5hmC."
    )
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv",
                        help="Path to DMR file. Supports CSV or BED (chr, start, end).")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--hm5c-bedgraph", nargs="+",
                        default=["/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"])
    parser.add_argument("--m5c-bedgraph", nargs="+", default=None,
                        help="Optional — not used by the model but accepted for symmetry with mainline scripts.")
    parser.add_argument("--atac-bw", nargs="+",
                        default=["/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw"])
    parser.add_argument("--sample-sizes", nargs="+", type=str, required=True)
    parser.add_argument("--chromosome", default=None)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--use-positional-encoding",
        action="store_true",
        help="Add sinusoidal positional encoding to query and context embeddings before CrossHyena.",
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--atac-scaling", choices=["none", "minmax"], default="minmax")
    add_clip_at_zero_argument(parser)
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (bfloat16). Recommended for long sequences.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Accumulate gradients over N mini-batches. Use with --batch-size 4-8 for long sequences.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade compute for memory. Recommended for very long sequences (8k+).")
    parser.add_argument("--mask-mode", choices=["cpg_both", "cpg_forward", "c_only", "all"], default="cpg_forward")
    parser.add_argument("--augment-reverse-complement", action="store_true",
                        help="Augment training data with reverse-complement views.")
    parser.add_argument("--lazy", action="store_true",
                        help="Required: fetch sequence/track data on-the-fly per batch.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible initialization and dataloader shuffling.")
    parser.add_argument("--output-csv", default="output/atac_query_sequence_context_results.csv")
    parser.add_argument("--output-json", default="output/atac_query_sequence_context_results.json")
    parser.add_argument("--timestamp", default="", help="Optional timestamp string for output path templates.")
    parser.add_argument(
        "--prediction-signal-h5ad",
        default="output/{timestamp}_atac_query_sequence_context_prediction_signals_{sample_size}.h5ad",
        help="Per-sample-size h5ad export path template for predicted and true methylation signals.",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="output/{timestamp}_atac_query_sequence_context_regression_plot_{sample_size}.png",
        help="Per-sample-size regression plot output path template.",
    )
    parser.add_argument(
        "--best-checkpoint-path",
        default="output/{timestamp}_atac_query_sequence_context_best_{sample_size}.pt",
    )
    parser.add_argument(
        "--last-checkpoint-path",
        default="output/{timestamp}_atac_query_sequence_context_last_{sample_size}.pt",
    )
    parser.set_defaults(use_m5c=False)
    return parser.parse_args()


def main():
    args = parse_args()
    args.sample_sizes = resolve_sample_sizes(args.sample_sizes, args)
    if args.chromosome is not None:
        from data import normalize_chromosome_label  # local import to avoid top-level churn
        args.chromosome = normalize_chromosome_label(args.chromosome)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args, lazy=getattr(args, "lazy", False))
    if args.chromosome is not None:
        # Re-use the chromosome filter from the mainline trainer.
        from run_m5c_query_sequence_atac_crosshyena_experiments import subset_dataset_by_chromosome
        df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = subset_dataset_by_chromosome(
            df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, args.chromosome
        )[1:]
    results = []
    for sample_size in args.sample_sizes:
        results.append(asdict(run_experiment(sample_size, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)))

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as file_obj:
        json.dump({"args": vars(args), "results": results}, file_obj, indent=2)


if __name__ == "__main__":
    main()