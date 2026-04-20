import argparse
import json
from dataclasses import asdict, dataclass

import pandas as pd
import torch
import torch.nn as nn

from data import CONTEXT_MODALITIES, TRACK_MODALITIES, load_data, prepare_modality_experiment_data
from models import MinimalCrossHyenaRegressor
from utils import export_prediction_signals, plot_regression_predictions


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


@dataclass
class ExperimentResult:
    num_dmrs: int
    input_modality: str
    context_modalities: str
    target_modality: str
    mask_mode: str
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


def get_context_dim(context_modalities: list[str]) -> int:
    context_dim = 0
    for modality in context_modalities:
        if modality == "sequence":
            context_dim += 4
        elif modality in TRACK_MODALITIES:
            context_dim += 1
        else:
            raise ValueError(f"Unknown context modality: {modality}")
    if context_dim == 0:
        raise ValueError("At least one context modality must be enabled.")
    return context_dim


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
    prepared = prepare_modality_experiment_data(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)

    model = MinimalCrossHyenaRegressor(
        seq_len=prepared.seq_len,
        query_dim=1,
        context_dim=get_context_dim(args.context_modalities),
        hidden_dim=args.hidden_dim,
        post_filter_len=prepared.post_filter_len,
        position_encoding=args.position_encoding,
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

    signal_csv = args.prediction_signal_csv.format(
        sample_size=prepared.usable_dmrs,
        input_modality=args.input_modality,
        target_modality=args.target_modality,
        timestamp=args.timestamp,
    )
    regression_plot = args.regression_plot_path.format(
        sample_size=prepared.usable_dmrs,
        input_modality=args.input_modality,
        target_modality=args.target_modality,
        timestamp=args.timestamp,
    )
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
        title=f"{args.target_modality} prediction from {args.input_modality} (n={prepared.usable_dmrs})",
    )

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        input_modality=args.input_modality,
        context_modalities=",".join(args.context_modalities),
        target_modality=args.target_modality,
        mask_mode=args.mask_mode,
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
        description="Run unified multimodal track-to-track experiments with sequence and signal context."
    )
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--m5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz")
    parser.add_argument("--hm5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz")
    parser.add_argument("--atac-bw", default="/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw")
    parser.add_argument("--sample-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--input-modality", choices=sorted(TRACK_MODALITIES), required=True)
    parser.add_argument("--target-modality", choices=sorted(TRACK_MODALITIES), required=True)
    parser.add_argument(
        "--context-modalities",
        nargs="+",
        choices=sorted(CONTEXT_MODALITIES),
        required=True,
        help="One or more context modalities. Use 'sequence' and/or any signal track.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["cpg_both", "cpg_forward", "all"],
        default="cpg_both",
        help="Loss mask to apply over positions.",
    )
    parser.add_argument(
        "--augment-reverse-complement",
        action="store_true",
        help="After train/val split, augment each subset with reverse-complement sequence views and reversed signal tracks.",
    )
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--position-encoding", choices=["none", "sinusoidal"], default="none")
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
    parser.add_argument("--output-csv", default="output/multimodal_track_results.csv")
    parser.add_argument("--output-json", default="output/multimodal_track_results.json")
    parser.add_argument("--timestamp", default="", help="Optional timestamp string for output path templates.")
    parser.add_argument(
        "--prediction-signal-csv",
        default="output/{timestamp}_{input_modality}_to_{target_modality}_{sample_size}.csv",
        help="Per-sample-size CSV export path template.",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="output/{timestamp}_{input_modality}_to_{target_modality}_{sample_size}.png",
        help="Per-sample-size regression plot output path template.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.context_modalities = list(dict.fromkeys(args.context_modalities))
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args)
    results = []
    for sample_size in args.sample_sizes:
        results.append(asdict(run_experiment(sample_size, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)))

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()