import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import (
    assign_non_overlapping_groups,
    find_cpg_candidate_positions,
    load_data,
    scale_atac_tensor,
    sequence_to_base_ids,
)
from models import MinimalCrossHyenaRegressor
from utils import export_prediction_signals, plot_regression_predictions


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


def prepare_atac_query_sequence_context_data(
    num_dmrs: int,
    args,
    df_dmr,
    seqs,
    _mcg_tracks,
    hmcg_tracks,
    atac_tracks,
) -> PreparedAtacSequenceData:
    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(hmcg_tracks), len(atac_tracks))
    seq_len = min(len(hmcg_tracks[0]), len(atac_tracks[0]), len(seqs[0]))
    post_filter_len = min(seq_len, 4)
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}

    query_tensor = torch.tensor(
        np.stack([np.asarray(atac_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
        dtype=torch.float32,
    ).unsqueeze(-1)
    query_tensor = scale_atac_tensor(query_tensor, args.atac_scaling)

    hm5c_target = torch.tensor(
        np.stack([np.asarray(hmcg_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
        dtype=torch.float32,
    ).unsqueeze(-1)

    base_ids_tensor = torch.stack([
        sequence_to_base_ids(seqs[idx], seq_len, base_to_index) for idx in range(usable_dmrs)
    ])
    sequence_onehot = F.one_hot(base_ids_tensor, num_classes=4).float()
    loss_mask = find_cpg_candidate_positions(base_ids_tensor).unsqueeze(-1).float()

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
        query_tensor[train_idx],
        sequence_onehot[train_idx],
        hm5c_target[train_idx],
        loss_mask[train_idx],
    )
    val_dataset = torch.utils.data.TensorDataset(
        query_tensor[val_idx],
        sequence_onehot[val_idx],
        hm5c_target[val_idx],
        loss_mask[val_idx],
    )

    return PreparedAtacSequenceData(
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        usable_dmrs=usable_dmrs,
        seq_len=seq_len,
        post_filter_len=post_filter_len,
        train_regions=len(train_idx),
        val_regions=len(val_idx),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
        val_region_metadata=split_regions_df.iloc[val_idx.numpy()].reset_index(drop=True),
    )


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    targets = []
    masks = []
    with torch.no_grad():
        for query_batch, context_batch, target_batch, mask_batch in loader:
            query_batch = query_batch.to(device)
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            pred = model(query_batch, context_batch)
            loss = masked_mse_loss(pred, target_batch, mask_batch)

            batch_count = query_batch.size(0)
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
        for query_batch, context_batch, target_batch, mask_batch in loader:
            query_batch = query_batch.to(device)
            context_batch = context_batch.to(device)
            pred = model(query_batch, context_batch)
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), torch.cat(masks, dim=0)


def run_experiment(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> ExperimentResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.num_epochs)

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_r2 = float("nan")
    best_val_pearsonr = float("nan")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for query_batch, context_batch, target_batch, mask_batch in prepared.train_loader:
            query_batch = query_batch.to(device)
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            pred = model(query_batch, context_batch)
            loss = masked_mse_loss(pred, target_batch, mask_batch)
            loss.backward()
            optimizer.step()

            batch_count = query_batch.size(0)
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
            patience_left = args.patience
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_loss, final_val_r2, final_val_pearsonr = evaluate(model, prepared.val_loader, device)
    final_preds, final_targets, final_masks = collect_predictions(model, prepared.val_loader, device)

    signal_csv = args.prediction_signal_csv.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    regression_plot = args.regression_plot_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    export_prediction_signals(
        signal_csv,
        prepared.val_region_metadata,
        final_preds.numpy(),
        final_targets.numpy(),
        final_masks.numpy(),
    )
    plot_regression_predictions(
        regression_plot,
        final_preds.numpy(),
        final_targets.numpy(),
        final_masks.numpy(),
        title=f"ATAC Query vs Sequence Context (n={prepared.usable_dmrs})",
    )

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        query_modality="atac",
        context_modality="sequence",
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
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run non-overlap ATAC-query and sequence-context experiments for predicting 5hmC."
    )
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--hm5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz")
    parser.add_argument("--atac-bw", default="/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw")
    parser.add_argument("--sample-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
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
    parser.add_argument("--output-csv", default="output/atac_query_sequence_context_results.csv")
    parser.add_argument("--output-json", default="output/atac_query_sequence_context_results.json")
    parser.add_argument("--timestamp", default="", help="Optional timestamp string for output path templates.")
    parser.add_argument(
        "--prediction-signal-csv",
        default="output/{timestamp}_atac_query_sequence_context_prediction_signals_{sample_size}.csv",
        help="Per-sample-size CSV export path template for predicted and true methylation signals.",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="output/{timestamp}_atac_query_sequence_context_regression_plot_{sample_size}.png",
        help="Per-sample-size regression plot output path template.",
    )
    parser.set_defaults(use_m5c=False)
    parser.add_argument("--m5c-bedgraph", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args)
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