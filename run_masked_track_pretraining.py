import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import (
    assign_non_overlapping_groups,
    build_sequence_tensor,
    generate_pretraining_cpg_mask,
    get_track_arrays,
    load_data,
    tensorize_track_modality,
)
from models import MaskedTrackPretrainingModelB
from utils import set_random_seed


TRACK_NAMES = ["5mc", "5hmc", "atac"]


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


@dataclass
class PreparedPretrainingData:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    usable_dmrs: int
    seq_len: int
    train_regions: int
    val_regions: int
    non_overlap_groups: int


def prepare_pretraining_data(
    num_dmrs: int,
    args,
    df_dmr,
    seqs,
    mcg_tracks,
    hmcg_tracks,
    atac_tracks,
) -> PreparedPretrainingData:
    if not mcg_tracks:
        raise ValueError("Pretraining requires 5mC tracks. Set --m5c-bedgraph to a valid path.")

    track_lengths = [len(hmcg_tracks[0]), len(atac_tracks[0]), len(seqs[0]), len(mcg_tracks[0])]
    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(mcg_tracks), len(hmcg_tracks), len(atac_tracks))
    seq_len = min(track_lengths)

    track_arrays = get_track_arrays(args, mcg_tracks, hmcg_tracks, atac_tracks, usable_dmrs, seq_len)
    base_ids_tensor, sequence_tensor = build_sequence_tensor(seqs, usable_dmrs, seq_len)
    m5c_tensor = tensorize_track_modality("5mc", track_arrays, args)
    h5mc_tensor = tensorize_track_modality("5hmc", track_arrays, args)
    atac_tensor = tensorize_track_modality("atac", track_arrays, args)

    split_regions_df = df_dmr.iloc[:usable_dmrs].copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
    split_regions_df = assign_non_overlapping_groups(split_regions_df, "chr", "start_expanded", "end_expanded")

    group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * args.train_ratio))
    train_group_ids = set(group_ids[:num_train_groups].tolist())
    train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).long()
    val_idx = torch.from_numpy(np.flatnonzero(~train_mask)).long()

    train_dataset = torch.utils.data.TensorDataset(
        m5c_tensor[train_idx],
        h5mc_tensor[train_idx],
        atac_tensor[train_idx],
        sequence_tensor[train_idx],
        base_ids_tensor[train_idx],
    )
    val_dataset = torch.utils.data.TensorDataset(
        m5c_tensor[val_idx],
        h5mc_tensor[val_idx],
        atac_tensor[val_idx],
        sequence_tensor[val_idx],
        base_ids_tensor[val_idx],
    )

    return PreparedPretrainingData(
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        usable_dmrs=usable_dmrs,
        seq_len=seq_len,
        train_regions=len(train_dataset),
        val_regions=len(val_dataset),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
    )


def sample_track_masks(base_ids_batch: torch.Tensor, mask_fraction: float) -> list[torch.Tensor]:
    shared_mask = generate_pretraining_cpg_mask(base_ids_batch, mask_fraction=mask_fraction, seed=None)
    return [shared_mask.clone() for _ in TRACK_NAMES]


def apply_masks_to_tracks(track_tensors: list[torch.Tensor], mask_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [track * (1.0 - mask) for track, mask in zip(track_tensors, mask_tensors)]


def compute_multitrack_masked_loss(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {}
    total_loss = 0.0
    for name, pred, target, mask in zip(TRACK_NAMES, preds, targets, masks):
        loss = masked_mse_loss(pred, target, mask)
        losses[name] = float(loss.detach().cpu().item())
        total_loss = total_loss + loss
    total_loss = total_loss / len(TRACK_NAMES)
    losses["total"] = float(total_loss.detach().cpu().item())
    return total_loss, losses


def evaluate(model: nn.Module, loader, device: torch.device, mask_fraction: float) -> dict[str, float]:
    model.eval()
    track_loss_sums = {name: 0.0 for name in TRACK_NAMES}
    total_loss_sum = 0.0
    seen = 0

    with torch.no_grad():
        for m5c_batch, h5mc_batch, atac_batch, sequence_batch, base_ids_batch in loader:
            m5c_batch = m5c_batch.to(device)
            h5mc_batch = h5mc_batch.to(device)
            atac_batch = atac_batch.to(device)
            sequence_batch = sequence_batch.to(device)

            masks = sample_track_masks(base_ids_batch, mask_fraction)
            masks = [mask.to(device) for mask in masks]

            original_tracks = [m5c_batch, h5mc_batch, atac_batch]
            masked_tracks = apply_masks_to_tracks(original_tracks, masks)
            preds = model(masked_tracks, sequence_batch)

            total_loss, losses = compute_multitrack_masked_loss(preds, original_tracks, masks)

            batch_size = m5c_batch.size(0)
            seen += batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            for name in TRACK_NAMES:
                track_loss_sums[name] += losses[name] * batch_size

    denom = max(seen, 1)
    result = {f"val_{name}_loss": track_loss_sums[name] / denom for name in TRACK_NAMES}
    result["val_total_loss"] = total_loss_sum / denom
    return result


@dataclass
class ExperimentResult:
    num_dmrs: int
    train_regions: int
    val_regions: int
    non_overlap_groups: int
    best_epoch: int
    final_lr: float
    best_val_total_loss: float
    best_val_5mc_loss: float
    best_val_5hmc_loss: float
    best_val_atac_loss: float
    final_val_total_loss: float
    final_val_5mc_loss: float
    final_val_5hmc_loss: float
    final_val_atac_loss: float
    checkpoint_paths: dict


def run_experiment(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> ExperimentResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_pretraining_data(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)

    model = MaskedTrackPretrainingModelB(
        seq_len=prepared.seq_len,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=args.use_positional_encoding,
        num_blocks=args.num_blocks,
        fusion_type=args.fusion_type,
    ).to(device)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.num_epochs)

    best_epoch = 0
    best_metrics = {
        "val_total_loss": float("inf"),
        "val_5mc_loss": float("inf"),
        "val_5hmc_loss": float("inf"),
        "val_atac_loss": float("inf"),
    }
    best_state = None
    last_epoch = 0

    best_checkpoint_path = args.best_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    last_checkpoint_path = args.last_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)

    patience_left = args.patience

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        model.train()

        train_track_loss_sums = {name: 0.0 for name in TRACK_NAMES}
        train_total_loss_sum = 0.0
        seen = 0

        for m5c_batch, h5mc_batch, atac_batch, sequence_batch, base_ids_batch in prepared.train_loader:
            m5c_batch = m5c_batch.to(device)
            h5mc_batch = h5mc_batch.to(device)
            atac_batch = atac_batch.to(device)
            sequence_batch = sequence_batch.to(device)

            masks = sample_track_masks(base_ids_batch, args.mask_fraction)
            masks = [mask.to(device) for mask in masks]

            original_tracks = [m5c_batch, h5mc_batch, atac_batch]
            masked_tracks = apply_masks_to_tracks(original_tracks, masks)

            optimizer.zero_grad()
            preds = model(masked_tracks, sequence_batch)
            total_loss, losses = compute_multitrack_masked_loss(preds, original_tracks, masks)
            total_loss.backward()
            optimizer.step()

            batch_size = m5c_batch.size(0)
            seen += batch_size
            train_total_loss_sum += float(total_loss.item()) * batch_size
            for name in TRACK_NAMES:
                train_track_loss_sums[name] += losses[name] * batch_size

        denom = max(seen, 1)
        train_total_loss = train_total_loss_sum / denom
        train_5mc_loss = train_track_loss_sums["5mc"] / denom
        train_5hmc_loss = train_track_loss_sums["5hmc"] / denom
        train_atac_loss = train_track_loss_sums["atac"] / denom

        val_metrics = evaluate(model, prepared.val_loader, device, args.mask_fraction)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["val_total_loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[num_dmrs={prepared.usable_dmrs}] Epoch {epoch:02d} | "
            f"train_total={train_total_loss:.4f} (5mC={train_5mc_loss:.4f}, 5hmC={train_5hmc_loss:.4f}, ATAC={train_atac_loss:.4f}) | "
            f"val_total={val_metrics['val_total_loss']:.4f} (5mC={val_metrics['val_5mc_loss']:.4f}, "
            f"5hmC={val_metrics['val_5hmc_loss']:.4f}, ATAC={val_metrics['val_atac_loss']:.4f}) | "
            f"lr={current_lr:.6g}"
        )

        if val_metrics["val_total_loss"] < best_metrics["val_total_loss"]:
            best_metrics = val_metrics.copy()
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                {
                    "train_total_loss": train_total_loss,
                    "train_5mc_loss": train_5mc_loss,
                    "train_5hmc_loss": train_5hmc_loss,
                    "train_atac_loss": train_atac_loss,
                    **val_metrics,
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

    final_val_metrics = evaluate(model, prepared.val_loader, device, args.mask_fraction)

    save_checkpoint(
        last_checkpoint_path,
        model,
        optimizer,
        scheduler,
        last_epoch,
        {
            **final_val_metrics,
            "is_best": last_epoch == best_epoch,
        },
        args,
        prepared.usable_dmrs,
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        final_val_metrics = evaluate(model, prepared.val_loader, device, args.mask_fraction)

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        train_regions=prepared.train_regions,
        val_regions=prepared.val_regions,
        non_overlap_groups=prepared.non_overlap_groups,
        best_epoch=best_epoch,
        final_lr=optimizer.param_groups[0]["lr"],
        best_val_total_loss=best_metrics["val_total_loss"],
        best_val_5mc_loss=best_metrics["val_5mc_loss"],
        best_val_5hmc_loss=best_metrics["val_5hmc_loss"],
        best_val_atac_loss=best_metrics["val_atac_loss"],
        final_val_total_loss=final_val_metrics["val_total_loss"],
        final_val_5mc_loss=final_val_metrics["val_5mc_loss"],
        final_val_5hmc_loss=final_val_metrics["val_5hmc_loss"],
        final_val_atac_loss=final_val_metrics["val_atac_loss"],
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised masked pretraining with 15% CpG masking per track "
            "(5mC, 5hmC, ATAC), reconstructing masked values."
        )
    )
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument(
        "--m5c-bedgraph",
        nargs="+",
        default=["/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz"],
    )
    parser.add_argument(
        "--hm5c-bedgraph",
        nargs="+",
        default=["/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"],
    )
    parser.add_argument(
        "--atac-bw",
        nargs="+",
        default=["/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw"],
    )
    parser.add_argument("--sample-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--chromosome", default=None)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument(
        "--fusion-type",
        choices=["cross_hyena", "cross_attention"],
        default="cross_hyena",
    )
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
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--use-all-input-groups", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", default="output/masked_track_pretraining_results.csv")
    parser.add_argument("--output-json", default="output/masked_track_pretraining_results.json")
    parser.add_argument("--timestamp", default="")
    parser.add_argument(
        "--best-checkpoint-path",
        default="output/{timestamp}_masked_track_pretraining_best_{sample_size}.pt",
    )
    parser.add_argument(
        "--last-checkpoint-path",
        default="output/{timestamp}_masked_track_pretraining_last_{sample_size}.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_m5c = True

    if not (0.0 < args.mask_fraction < 1.0):
        raise ValueError("--mask-fraction must be in (0, 1).")

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
