import argparse
import json
from dataclasses import dataclass, asdict

import pandas as pd
import torch
import torch.nn as nn
 
from data import load_data, prepare_experiment_data
from models import MinimalCrossHyenaRegressor
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


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    targets = []
    masks = []
    with torch.no_grad():
        for query_batch, sequence_batch, atac_batch, target_batch, mask_batch in loader:
            query_batch = query_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            context_batch = build_context_batch(sequence_batch, atac_batch, model.run_args)
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
        for query_batch, sequence_batch, atac_batch, target_batch, mask_batch in loader:
            query_batch = query_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            context_batch = build_context_batch(sequence_batch, atac_batch, model.run_args)
            pred = model(query_batch, context_batch)
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), torch.cat(masks, dim=0)


@dataclass
class ExperimentResult:
    num_dmrs: int
    use_m5c: bool
    use_sequence: bool
    use_atac: bool
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


def build_context_batch(sequence_batch: torch.Tensor, atac_batch: torch.Tensor, args) -> torch.Tensor:
    context_parts = []
    if args.use_sequence:
        context_parts.append(sequence_batch)
    if args.use_atac:
        context_parts.append(atac_batch)
    if not context_parts:
        raise ValueError("At least one context modality must be enabled.")
    return torch.cat(context_parts, dim=-1)


def get_context_dim(args) -> int:
    context_dim = 0
    if args.use_sequence:
        context_dim += 4
    if args.use_atac:
        context_dim += 1
    if context_dim == 0:
        raise ValueError("At least one context modality must be enabled.")
    return context_dim


def run_experiment(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> ExperimentResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_experiment_data(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)

    model = MinimalCrossHyenaRegressor(
        seq_len=prepared.seq_len,
        query_dim=1,
        context_dim=get_context_dim(args),
        hidden_dim=args.hidden_dim,
        post_filter_len=prepared.post_filter_len,
        use_positional_encoding=args.use_positional_encoding,
    ).to(device)
    model.run_args = args
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
        for query_batch, sequence_batch, atac_batch, target_batch, mask_batch in prepared.train_loader:
            query_batch = query_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            context_batch = build_context_batch(sequence_batch, atac_batch, args)

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        title=f"Ground Truth vs Prediction (n={prepared.usable_dmrs})",
    )

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        use_m5c=args.use_m5c,
        use_sequence=args.use_sequence,
        use_atac=args.use_atac,
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
    parser = argparse.ArgumentParser(description="Run non-overlap 5mC+Sequence+ATAC to 5hmC experiments at different sample sizes.")
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--m5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz")
    parser.add_argument("--hm5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz")
    parser.add_argument("--atac-bw", default="/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw")
    parser.set_defaults(use_m5c=True)
    parser.add_argument("--use-m5c", dest="use_m5c", action="store_true", help="Use the 5mC track as the query input.")
    parser.add_argument("--no-use-m5c", dest="use_m5c", action="store_false", help="Disable the 5mC query input and replace it with zeros.")
    parser.set_defaults(use_sequence=True, use_atac=True)
    parser.add_argument("--use-sequence", dest="use_sequence", action="store_true", help="Use DNA sequence in the context input.")
    parser.add_argument("--no-use-sequence", dest="use_sequence", action="store_false", help="Disable DNA sequence in the context input.")
    parser.add_argument("--use-atac", dest="use_atac", action="store_true", help="Use ATAC signal in the context input.")
    parser.add_argument("--no-use-atac", dest="use_atac", action="store_false", help="Disable ATAC signal in the context input.")
    parser.add_argument("--sample-sizes", nargs="+", type=int, required=True)
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
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible initialization and dataloader shuffling.")
    parser.add_argument("--output-csv", default="output/sample_size_results.csv")
    parser.add_argument("--output-json", default="output/sample_size_results.json")
    parser.add_argument("--timestamp", default="", help="Optional timestamp string for output path templates.")
    parser.add_argument(
        "--prediction-signal-csv",
        default="output/{timestamp}_prediction_signals_{sample_size}.csv",
        help="Per-sample-size CSV export path template for predicted and true methylation signals.",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="output/{timestamp}_regression_plot_{sample_size}.png",
        help="Per-sample-size regression plot output path template.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args)
    results = []
    for sample_size in args.sample_sizes:
        results.append(asdict(run_experiment(sample_size, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)))

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()