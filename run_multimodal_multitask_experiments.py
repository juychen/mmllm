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
from data import augment_with_reverse_complement, prepare_multimodal_multitask_data
from models import MinimalCrossHyenaRegressor
from utils import export_prediction_signals, plot_regression_predictions, set_random_seed

@dataclass
class ExperimentResult:
    num_dmrs: int
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
    signal_csvs: dict
    regression_plots: dict
    final_val_loss_per_task: dict
    final_val_r2_per_task: dict
    final_val_pearsonr_per_task: dict

def multitask_masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, task_weights=None) -> torch.Tensor:
    T = pred.shape[-1]
    if task_weights is None:
        task_weights = torch.ones(T, device=pred.device)
    per_task_losses = []
    for t in range(T):
        se = (pred[..., t] - target[..., t]).pow(2)
        m = mask[..., t]
        loss_t = (se * m).sum() / m.sum().clamp_min(1.0)
        per_task_losses.append(loss_t)
    per_task_losses = torch.stack(per_task_losses)
    task_weights = task_weights.to(per_task_losses.device)
    weighted = (per_task_losses * task_weights).sum() / task_weights.sum().clamp_min(1.0)
    return weighted



def build_scheduler(optimizer, args, steps_per_experiment):
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

def build_optimizer(model, args):
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

def evaluate(model, loader, device, task_weights=None):
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
            loss = multitask_masked_loss(pred, target_batch.squeeze(-1), mask_batch, task_weights)
            batch_count = query_batch.size(0)
            total_loss += loss.item() * batch_count
            total_count += batch_count
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)
    ss_res = (((targets.squeeze(-1) - preds) ** 2) * masks).sum()
    masked_targets = targets.squeeze(-1)[masks.bool()]
    masked_preds = preds[masks.bool()]
    target_mean = masked_targets.mean() if masked_targets.numel() > 0 else torch.tensor(0.0)
    ss_tot = (((targets.squeeze(-1) - target_mean) ** 2) * masks).sum().clamp_min(1e-12)
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

def collect_predictions(model, loader, device):
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


def evaluate_per_task(model, loader, device):
    """Return per-task (loss, r2, pearson) as lists in task order."""
    preds, targets, masks = collect_predictions(model, loader, device)
    # preds: (N, L, T) or (N, T) depending on shape; ensure last dim is tasks
    # targets: may have extra last dim; squeeze if needed
    targets = targets.squeeze(-1)
    T = preds.shape[-1]
    losses = []
    r2s = []
    pearsons = []
    for t in range(T):
        pred_t = preds[..., t]
        target_t = targets[..., t]
        mask_t = masks[..., t]
        # MSE loss over masked positions
        se = (pred_t - target_t) ** 2
        denom = mask_t.sum().clamp_min(1.0)
        loss_t = (se * mask_t).sum() / denom
        losses.append(float(loss_t))
        # R2
        masked_targets = target_t[mask_t.bool()]
        masked_preds = pred_t[mask_t.bool()]
        if masked_targets.numel() > 0:
            target_mean = masked_targets.mean()
            ss_res = ((masked_targets - masked_preds) ** 2).sum()
            ss_tot = ((masked_targets - target_mean) ** 2).sum().clamp_min(1e-12)
            r2_t = 1.0 - (ss_res / ss_tot)
            r2s.append(float(r2_t))
        else:
            r2s.append(float('nan'))
        # Pearson
        if masked_targets.numel() > 1:
            centered_targets = masked_targets - masked_targets.mean()
            centered_preds = masked_preds - masked_preds.mean()
            denom = centered_targets.pow(2).sum().sqrt() * centered_preds.pow(2).sum().sqrt()
            pearson = (centered_targets * centered_preds).sum() / denom.clamp_min(1e-12)
            pearsons.append(float(pearson))
        else:
            pearsons.append(float('nan'))
    return losses, r2s, pearsons

def run_experiment(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_multimodal_multitask_data(
        num_dmrs,
        args,
        df_dmr,
        seqs,
        mcg_tracks,
        hmcg_tracks,
        atac_tracks,
    )
    model = MinimalCrossHyenaRegressor(
        seq_len=prepared["seq_len"],
        query_dim=1,
        context_dim=4,
        hidden_dim=args.hidden_dim,
        post_filter_len=prepared["post_filter_len"],
        use_positional_encoding=args.use_positional_encoding,
    ).to(device)
    # predict three channels: 5mc, 5hmc, c (complement), and convert to probabilities via softmax
    model.head = nn.Linear(args.hidden_dim, 3).to(device)
    task_weights = torch.tensor(args.task_weights, dtype=torch.float32) if args.task_weights is not None else None
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
        for query_batch, context_batch, target_batch, mask_batch in prepared["train_loader"]:
            query_batch = query_batch.to(device)
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            optimizer.zero_grad()
            pred = model(query_batch, context_batch)
            loss = multitask_masked_loss(pred, target_batch.squeeze(-1), mask_batch, task_weights)
            loss.backward()
            optimizer.step()
            batch_count = query_batch.size(0)
            running_loss += loss.item() * batch_count
            seen += batch_count
        train_loss = running_loss / max(seen, 1)
        val_loss, val_r2, val_pearsonr = evaluate(model, prepared["val_loader"], device, task_weights)
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[num_dmrs={prepared['usable_dmrs']}] Epoch {epoch:02d} | "
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
    final_val_loss, final_val_r2, final_val_pearsonr = evaluate(model, prepared["val_loader"], device, task_weights)
    final_preds, final_targets, final_masks = collect_predictions(model, prepared["val_loader"], device)
    # compute per-task final metrics
    per_losses, per_r2s, per_pearsons = evaluate_per_task(model, prepared["val_loader"], device)
    # map task names to per-task metrics
    task_names = ["5mc", "5hmc", "c"]
    final_val_loss_per_task = {task: per_losses[i] for i, task in enumerate(task_names)}
    final_val_r2_per_task = {task: per_r2s[i] for i, task in enumerate(task_names)}
    final_val_pearsonr_per_task = {task: per_pearsons[i] for i, task in enumerate(task_names)}
    # Base templates (used to derive per-task paths)
    signal_csv = args.prediction_signal_csv.format(sample_size=prepared["usable_dmrs"], timestamp=args.timestamp)
    regression_plot = args.regression_plot_path.format(sample_size=prepared["usable_dmrs"], timestamp=args.timestamp)
    # Export predictions and plots per task dimension. dim 0 -> 5mc, dim 1 -> 5hmc
    task_names = ["5mc", "5hmc", "c"]
    signal_csvs = {}
    regression_plots = {}
    for dim, task in enumerate(task_names):
        preds_dim = final_preds[..., dim].numpy()
        targets_dim = final_targets[..., dim].numpy()
        masks_dim = final_masks[..., dim].numpy()
        # create per-task file paths by inserting task name before extension
        if signal_csv.endswith(".csv"):
            #task_signal_csv = signal_csv.replace("multi_", f"multi3_{task}_")
            task_signal_csv = args.prediction_signal_csv.format(
                sample_size=f"{task}_{prepared['usable_dmrs']}",
                timestamp=args.timestamp,
            )

        else:
            task_signal_csv = f"{signal_csv}_{task}.csv"
        if regression_plot.endswith(".png"):
            #task_plot = regression_plot.replace("multi_", f"multi3_{task}_")
            task_plot = args.regression_plot_path.format(
                sample_size=f"{task}_{prepared['usable_dmrs']}",
                timestamp=args.timestamp,
            )
        else:
            task_plot = f"{regression_plot}_{task}.png"

        export_prediction_signals(
            task_signal_csv,
            prepared["val_region_metadata"],
            preds_dim,
            targets_dim,
            masks_dim,
        )
        plot_regression_predictions(
            task_plot,
            preds_dim,
            targets_dim,
            masks_dim,
            title=f"ATAC+sequence multitask {task} (n={prepared['usable_dmrs']})",
        )
        signal_csvs[task] = task_signal_csv
        regression_plots[task] = task_plot
    return ExperimentResult(
        num_dmrs=prepared["usable_dmrs"],
        train_regions=prepared["train_regions"],
        val_regions=prepared["val_regions"],
        non_overlap_groups=prepared["non_overlap_groups"],
        best_epoch=best_epoch,
        final_lr=optimizer.param_groups[0]["lr"],
        best_val_loss=best_val_loss,
        best_val_r2=best_val_r2,
        best_val_pearsonr=best_val_pearsonr,
        final_val_loss=final_val_loss,
        final_val_r2=final_val_r2,
        final_val_pearsonr=final_val_pearsonr,
        signal_csvs=signal_csvs,
        regression_plots=regression_plots,
        final_val_loss_per_task=final_val_loss_per_task,
        final_val_r2_per_task=final_val_r2_per_task,
        final_val_pearsonr_per_task=final_val_pearsonr_per_task,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ATAC+sequence multitask experiments predicting both 5mc and 5hmc."
    )
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--m5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz")
    parser.add_argument("--hm5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz")
    parser.add_argument("--atac-bw", default="/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw")
    parser.add_argument("--sample-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--input-modality", default='atac', required=False)
    parser.add_argument("--target-modality", default='multi', required=False)

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
    parser.add_argument("--output-csv", default="output/multimodal_multitask_results.csv")
    parser.add_argument("--output-json", default="output/multimodal_multitask_results.json")
    parser.add_argument("--timestamp", default="", help="Optional timestamp string for output path templates.")
    parser.add_argument(
        "--prediction-signal-csv",
        default="output/{timestamp}_multimodal_multitask_prediction_signals_{sample_size}.csv",
        help="Per-sample-size CSV export path template for predicted and true methylation signals.",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="output/{timestamp}_multimodal_multitask_regression_plot_{sample_size}.png",
        help="Per-sample-size regression plot output path template.",
    )
    parser.add_argument(
        "--task-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-task weights for multitask loss (space-separated floats, length=3).",
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
    return parser.parse_args()

def main():
    args = parse_args()
    # this script uses sequence as the only context modality
    args.context_modalities = ["sequence"]
    set_random_seed(args.seed)
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args)
    results = []
    for sample_size in args.sample_sizes:
        results.append(asdict(run_experiment(sample_size, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)))
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as file_obj:
        json.dump({"args": vars(args), "results": results}, file_obj, indent=2)

    # Also write per-modality JSON files (one per predicted task/modalitiy)
    task_names = ["5mc", "5hmc", "c"]
    for task in task_names:
        mod_results = []
        for r in results:
            mod_r = r.copy()
            signal = mod_r.pop("signal_csvs", {}).get(task)
            plot = mod_r.pop("regression_plots", {}).get(task)
            mod_r["signal_csv"] = signal
            mod_r["regression_plot"] = plot
            # replace aggregated final metrics with per-task metrics for this modality
            mod_r["final_val_loss"] = mod_r.pop("final_val_loss_per_task", {}).get(task)
            mod_r["final_val_r2"] = mod_r.pop("final_val_r2_per_task", {}).get(task)
            mod_r["final_val_pearsonr"] = mod_r.pop("final_val_pearsonr_per_task", {}).get(task)
            mod_results.append(mod_r)
        if args.output_json.endswith("_results.json"):
            mod_path = args.output_json.replace("_results.json", f"_{task}_results.json")
        else:
            mod_path = f"{args.output_json}_{task}.json"
        with open(mod_path, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "results": mod_results}, fh, indent=2)

if __name__ == "__main__":
    main()
