import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import (
    BASE_COMPLEMENT_INDEX,
    assign_non_overlapping_groups,
    build_sequence_tensor,
    get_track_arrays,
    load_data,
    resolve_loss_mask,
    tensorize_track_modality,
)
from models import M5CQuerySequenceAtacCrossHyenaRegressor
from utils import export_prediction_signals, plot_regression_predictions, set_random_seed


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


def save_checkpoint(checkpoint_path: str, model, optimizer, scheduler, epoch: int, metrics: dict, args, num_dmrs: int):
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


def reverse_complement_sequence_tensor(sequence_tensor: torch.Tensor) -> torch.Tensor:
    complement_index = BASE_COMPLEMENT_INDEX.to(sequence_tensor.device)
    complemented = sequence_tensor.index_select(dim=-1, index=complement_index)
    return torch.flip(complemented, dims=[1])


def augment_three_modalities(
    m5c_tensor: torch.Tensor,
    sequence_tensor: torch.Tensor,
    atac_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    loss_mask: torch.Tensor,
    base_ids_tensor: torch.Tensor,
    region_metadata: pd.DataFrame,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    if not getattr(args, "augment_reverse_complement", False):
        metadata = region_metadata.copy().reset_index(drop=True)
        metadata["strand_view"] = "+"
        return m5c_tensor, sequence_tensor, atac_tensor, target_tensor, loss_mask, metadata

    rc_m5c_tensor = torch.flip(m5c_tensor, dims=[1])
    rc_sequence_tensor = reverse_complement_sequence_tensor(sequence_tensor)
    rc_atac_tensor = torch.flip(atac_tensor, dims=[1])
    rc_target_tensor = torch.flip(target_tensor, dims=[1])
    rc_base_ids_tensor = torch.argmax(rc_sequence_tensor, dim=-1)
    rc_loss_mask = resolve_loss_mask(args.mask_mode, rc_base_ids_tensor)

    forward_metadata = region_metadata.copy().reset_index(drop=True)
    forward_metadata["strand_view"] = "+"
    rc_metadata = region_metadata.copy().reset_index(drop=True)
    rc_metadata["strand_view"] = "-"
    augmented_metadata = pd.concat([forward_metadata, rc_metadata], ignore_index=True)

    return (
        torch.cat([m5c_tensor, rc_m5c_tensor], dim=0),
        torch.cat([sequence_tensor, rc_sequence_tensor], dim=0),
        torch.cat([atac_tensor, rc_atac_tensor], dim=0),
        torch.cat([target_tensor, rc_target_tensor], dim=0),
        torch.cat([loss_mask, rc_loss_mask], dim=0),
        augmented_metadata,
    )


@dataclass
class PreparedSequenceAtacData:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    usable_dmrs: int
    seq_len: int
    post_filter_len: int
    train_regions: int
    val_regions: int
    non_overlap_groups: int
    val_region_metadata: pd.DataFrame


def prepare_sequence_atac_crosshyena_data(
    num_dmrs: int,
    args,
    df_dmr,
    seqs,
    mcg_tracks,
    hmcg_tracks,
    atac_tracks,
) -> PreparedSequenceAtacData:
    if not mcg_tracks:
        raise ValueError("M5CQuerySequenceAtacCrossHyenaRegressor requires 5mC tracks. Set --m5c-bedgraph to a valid path.")

    track_lengths = [len(hmcg_tracks[0]), len(atac_tracks[0]), len(seqs[0]), len(mcg_tracks[0])]
    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(mcg_tracks), len(hmcg_tracks), len(atac_tracks))
    seq_len = min(track_lengths)
    post_filter_len = min(seq_len, 4)

    track_arrays = get_track_arrays(args, mcg_tracks, hmcg_tracks, atac_tracks, usable_dmrs, seq_len)
    base_ids_tensor, sequence_tensor = build_sequence_tensor(seqs, usable_dmrs, seq_len)
    m5c_tensor = tensorize_track_modality("5mc", track_arrays, args)
    atac_tensor = tensorize_track_modality("atac", track_arrays, args)
    target_tensor = tensorize_track_modality("5hmc", track_arrays, args)
    loss_mask = resolve_loss_mask(args.mask_mode, base_ids_tensor)

    split_regions_df = df_dmr.iloc[:usable_dmrs].copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
    split_regions_df["sequence"] = [str(seqs[idx])[:seq_len].upper() for idx in range(usable_dmrs)]
    split_regions_df = assign_non_overlapping_groups(split_regions_df, "chr", "start_expanded", "end_expanded")

    group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * args.train_ratio))
    train_group_ids = set(group_ids[:num_train_groups].tolist())
    train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).long()
    val_idx = torch.from_numpy(np.flatnonzero(~train_mask)).long()

    train_region_metadata = split_regions_df.iloc[train_idx.numpy()].reset_index(drop=True)
    val_region_metadata = split_regions_df.iloc[val_idx.numpy()].reset_index(drop=True)

    train_m5c_tensor, train_sequence_tensor, train_atac_tensor, train_target_tensor, train_loss_mask, train_region_metadata = augment_three_modalities(
        m5c_tensor[train_idx],
        sequence_tensor[train_idx],
        atac_tensor[train_idx],
        target_tensor[train_idx],
        loss_mask[train_idx],
        base_ids_tensor[train_idx],
        train_region_metadata,
        args,
    )
    val_m5c_tensor, val_sequence_tensor, val_atac_tensor, val_target_tensor, val_loss_mask, val_region_metadata = augment_three_modalities(
        m5c_tensor[val_idx],
        sequence_tensor[val_idx],
        atac_tensor[val_idx],
        target_tensor[val_idx],
        loss_mask[val_idx],
        base_ids_tensor[val_idx],
        val_region_metadata,
        args,
    )

    train_dataset = torch.utils.data.TensorDataset(
        train_m5c_tensor,
        train_sequence_tensor,
        train_atac_tensor,
        train_target_tensor,
        train_loss_mask,
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_m5c_tensor,
        val_sequence_tensor,
        val_atac_tensor,
        val_target_tensor,
        val_loss_mask,
    )

    return PreparedSequenceAtacData(
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        usable_dmrs=usable_dmrs,
        seq_len=seq_len,
        post_filter_len=post_filter_len,
        train_regions=len(train_dataset),
        val_regions=len(val_dataset),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
        val_region_metadata=val_region_metadata,
    )


def subset_dataset_by_chromosome(df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks, chromosome: str):
    normalized_chromosome = normalize_chromosome_label(chromosome)
    chrom_series = df_dmr["chr"].astype(str).map(normalize_chromosome_label)
    matched_indices = np.flatnonzero(chrom_series == normalized_chromosome)
    if matched_indices.size == 0:
        available_chromosomes = sorted(chrom_series.drop_duplicates().tolist())
        raise ValueError(
            f"No DMR rows found for chromosome {normalized_chromosome}. "
            f"Available chromosomes: {', '.join(available_chromosomes)}"
        )
    filtered_df = df_dmr.iloc[matched_indices].reset_index(drop=True)
    index_list = matched_indices.tolist()
    filtered_seqs = [seqs[idx] for idx in index_list]
    filtered_mcg_tracks = [mcg_tracks[idx] for idx in index_list]
    filtered_hmcg_tracks = [hmcg_tracks[idx] for idx in index_list]
    filtered_atac_tracks = [atac_tracks[idx] for idx in index_list]
    return normalized_chromosome, filtered_df, filtered_seqs, filtered_mcg_tracks, filtered_hmcg_tracks, filtered_atac_tracks


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


def run_experiment(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> ExperimentResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_sequence_atac_crosshyena_data(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)

    model = M5CQuerySequenceAtacCrossHyenaRegressor(
        seq_len=prepared.seq_len,
        hidden_dim=args.hidden_dim,
        post_filter_len=prepared.post_filter_len,
        use_positional_encoding=args.use_positional_encoding,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.num_epochs)

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_r2 = float("nan")
    best_val_pearsonr = float("nan")
    best_state = None
    last_epoch = 0
    best_checkpoint_path = args.best_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    last_checkpoint_path = args.last_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    patience_left = args.patience

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        model.train()
        running_loss = 0.0
        seen = 0
        for m5c_batch, sequence_batch, atac_batch, target_batch, mask_batch in prepared.train_loader:
            m5c_batch = m5c_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            pred = model(m5c_batch, sequence_batch, atac_batch)
            loss = masked_mse_loss(pred, target_batch, mask_batch)
            loss.backward()
            optimizer.step()

            batch_count = m5c_batch.size(0)
            running_loss += loss.item() * batch_count
            seen += batch_count

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
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "val_pearsonr": val_pearsonr,
                    "is_best": True,
                },
                args,
                prepared.usable_dmrs,
            )
            patience_left = args.patience
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                break

    save_checkpoint(
        last_checkpoint_path,
        model,
        optimizer,
        scheduler,
        last_epoch,
        {
            "val_loss": val_loss,
            "val_r2": val_r2,
            "val_pearsonr": val_pearsonr,
            "is_best": last_epoch == best_epoch,
        },
        args,
        prepared.usable_dmrs,
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_r2, final_val_pearsonr = evaluate(model, prepared.val_loader, device)
    final_preds, final_targets, final_masks = collect_predictions(model, prepared.val_loader, device)

    signal_csv = args.prediction_signal_csv.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    regression_plot = args.regression_plot_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
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
        title=f"5mC-query sequence+ATAC context -> 5hmC (n={prepared.usable_dmrs})",
    )

    input_group_files = []
    if "input_group" in df_dmr.columns:
        group_columns = [column for column in ["m5c_bedgraph_path", "hm5c_bedgraph_path", "atac_bw_path"] if column in df_dmr.columns]
        grouped_inputs = df_dmr.loc[:, ["input_group", *group_columns]].drop_duplicates().sort_values("input_group")
        for input_group, group_rows in grouped_inputs.groupby("input_group", sort=True):
            group_record = {"input_group": int(input_group)}
            if "m5c_bedgraph_path" in group_rows.columns:
                group_record["m5c_bedgraph"] = str(group_rows["m5c_bedgraph_path"].iloc[0])
            if "hm5c_bedgraph_path" in group_rows.columns:
                group_record["hm5c_bedgraph"] = str(group_rows["hm5c_bedgraph_path"].iloc[0])
            if "atac_bw_path" in group_rows.columns:
                group_record["atac_bw"] = str(group_rows["atac_bw_path"].iloc[0])
            input_group_files.append(group_record)
    else:
        input_group_files.append(
            {
                "input_group": 0,
                "m5c_bedgraph": args.m5c_bedgraph[0] if getattr(args, "m5c_bedgraph", None) else None,
                "hm5c_bedgraph": args.hm5c_bedgraph[0] if getattr(args, "hm5c_bedgraph", None) else None,
                "atac_bw": args.atac_bw[0] if getattr(args, "atac_bw", None) else None,
            }
        )

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        chromosome=getattr(args, "chromosome", None),
        input_group_files=input_group_files,
        output_files={
            "results_csv": args.output_csv,
            "results_json": args.output_json,
            "signal_csv": signal_csv,
            "regression_plot": regression_plot,
            "checkpoints": {
                "best": best_checkpoint_path,
                "last": last_checkpoint_path,
            },
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
        checkpoint_paths={
            "best": best_checkpoint_path,
            "last": last_checkpoint_path,
        },
    )


def normalize_chromosome_label(chromosome: str) -> str:
    chrom_value = str(chromosome).strip()
    if not chrom_value:
        raise ValueError("--chromosome cannot be empty.")
    return chrom_value if chrom_value.lower().startswith("chr") else f"chr{chrom_value}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 5mC-query sequence+ATAC-context -> 5hmC experiments with M5CQuerySequenceAtacCrossHyenaRegressor."
    )
    parser.add_argument("--dmr-csv", default="/data2st2/junyi/dmr_with_sequences.csv")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument(
        "--m5c-bedgraph",
        nargs="+",
        default=["/data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.m.bedGraph.gz"],
    )
    parser.add_argument(
        "--hm5c-bedgraph",
        nargs="+",
        default=["/data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.h.bedGraph.gz"],
    )
    parser.add_argument(
        "--atac-bw",
        nargs="+",
        default=["/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/PFC_MC_track.bw"],
    )
    parser.add_argument("--sample-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--chromosome", default=None)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
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
    parser.add_argument("--atac-scaling", choices=["none", "minmax"], default="minmax")
    parser.add_argument("--mask-mode", choices=["cpg_both", "cpg_forward", "all"], default="cpg_both")
    parser.add_argument("--augment-reverse-complement", action="store_true")
    parser.add_argument("--use-all-input-groups", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", default="/data1st2/zhangyr/data/mmllm/multimodal_test/PFC_MC/m5c_query_sequence_atac_crosshyena_results.csv")
    parser.add_argument("--output-json", default="/data1st2/zhangyr/data/mmllm/multimodal_test/PFC_MC/m5c_query_sequence_atac_crosshyena_results.json")
    parser.add_argument("--timestamp", default="")
    parser.add_argument(
        "--prediction-signal-csv",
        default="/data1st2/zhangyr/data/mmllm/multimodal_test/PFC_MC/{timestamp}_m5c_query_sequence_atac_crosshyena_prediction_signals_{sample_size}.csv",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="/data1st2/zhangyr/data/mmllm/multimodal_test/PFC_MC/{timestamp}_m5c_query_sequence_atac_crosshyena_regression_plot_{sample_size}.png",
    )
    parser.add_argument(
        "--best-checkpoint-path",
        default="/data1st2/zhangyr/data/mmllm/multimodal_test/PFC_MC/{timestamp}_m5c_query_sequence_atac_crosshyena_best_{sample_size}.pt",
    )
    parser.add_argument(
        "--last-checkpoint-path",
        default="/data1st2/zhangyr/data/mmllm/multimodal_test/PFC_MC/{timestamp}_m5c_query_sequence_atac_crosshyena_last_{sample_size}.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_m5c = True
    if args.chromosome is not None:
        args.chromosome = normalize_chromosome_label(args.chromosome)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args)
    if args.chromosome is not None:
        df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = subset_dataset_by_chromosome(
            df_dmr,
            seqs,
            mcg_tracks,
            hmcg_tracks,
            atac_tracks,
            args.chromosome,
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