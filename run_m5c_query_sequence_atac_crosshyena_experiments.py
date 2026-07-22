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
    BASE_COMPLEMENT_INDEX,
    LazyM5cSequenceAtacDataset,
    assign_non_overlapping_groups,
    ensure_path_list,
    get_sequence,
    load_data,
    resolve_loss_mask,
    write_train_val_beds,
)
from models import M5CQuerySequenceAtacCrossHyenaRegressor, M5CQuerySequenceAtacCrossHyenaRegressorModelB
from utils import (
    export_prediction_signals_h5ad,
    get_freest_gpu,
    plot_regression_predictions,
    resolve_sample_sizes,
    set_random_seed,
)


def transfer_pretrained_weights(
    model: M5CQuerySequenceAtacCrossHyenaRegressorModelB,
    pretrained_path: str,
    device: torch.device,
) -> None:
    """Transfer weights from MaskedTrackPretrainingModelB checkpoint to downstream model.

    Mapping detail:
      query_proj/query_norm        ← query_proj/query_norm
      sequence_proj/sequence_norm  ← sequence_proj/sequence_norm
      atac_proj/atac_norm          ← context_track_projs.1 / context_track_norms.1
      context_proj                 ← context_proj (slice: drop middle 5hmC third)
      context_norm                 ← context_norm
      blocks.i.*                   ← blocks.i.* (first N blocks)
      final_norm                   ← final_norm
      head.*                       ← heads.1.* (5hmC prediction head)
    """
    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained = checkpoint["model_state_dict"]
    model_state = model.state_dict()

    # Direct 1-to-1 mappings
    direct_pairs = [
        ("query_proj.weight", "query_proj.weight"),
        ("query_proj.bias", "query_proj.bias"),
        ("query_norm.weight", "query_norm.weight"),
        ("query_norm.bias", "query_norm.bias"),
        ("sequence_proj.weight", "sequence_proj.weight"),
        ("sequence_proj.bias", "sequence_proj.bias"),
        ("sequence_norm.weight", "sequence_norm.weight"),
        ("sequence_norm.bias", "sequence_norm.bias"),
        ("atac_proj.weight", "context_track_projs.1.weight"),
        ("atac_proj.bias", "context_track_projs.1.bias"),
        ("atac_norm.weight", "context_track_norms.1.weight"),
        ("atac_norm.bias", "context_track_norms.1.bias"),
        ("context_norm.weight", "context_norm.weight"),
        ("context_norm.bias", "context_norm.bias"),
        ("final_norm.weight", "final_norm.weight"),
        ("final_norm.bias", "final_norm.bias"),
        ("head.0.weight", "heads.1.0.weight"),
        ("head.0.bias", "heads.1.0.bias"),
        ("head.1.weight", "heads.1.1.weight"),
        ("head.1.bias", "heads.1.1.bias"),
        ("head.2.weight", "heads.1.2.weight"),
        ("head.2.bias", "heads.1.2.bias"),
    ]

    loaded_keys = set()
    for dst_key, src_key in direct_pairs:
        if src_key in pretrained and dst_key in model_state:
            model_state[dst_key].copy_(pretrained[src_key])
            loaded_keys.add(src_key)

    # context_proj.weight: pretrain has (hidden, (1+num_context_tracks)*hidden)
    #   num_context_tracks=1: [seq | 5hmc]            → (hidden, 2*hidden) → copy directly
    #   num_context_tracks=2: [seq | 5hmc | atac]     → (hidden, 3*hidden) → drop middle 5hmc
    # downstream needs (hidden, 2*hidden) = [seq | atac]
    if "context_proj.weight" in pretrained and "context_proj.weight" in model_state:
        pretrained_w = pretrained["context_proj.weight"]  # (hidden, pretrain_in_dim)
        downstream_w = model_state["context_proj.weight"]  # (hidden, 2*hidden)
        if pretrained_w.size(1) == downstream_w.size(1):
            # Same input dimension — copy directly
            model_state["context_proj.weight"].copy_(pretrained_w)
        elif pretrained_w.size(1) == 3 * downstream_w.size(0):
            # Pretrain has 3 parts: [seq | 5hmc | atac] — drop the middle 5hmc part
            hidden_dim = downstream_w.size(0)
            seq_part = pretrained_w[:, :hidden_dim]
            atac_part = pretrained_w[:, 2 * hidden_dim:]  # skip 5hmc in the middle
            model_state["context_proj.weight"].copy_(torch.cat([seq_part, atac_part], dim=1))
        else:
            print(
                f"[transfer_pretrained_weights] WARNING: context_proj.weight shape mismatch "
                f"pretrained={pretrained_w.shape}, downstream={downstream_w.shape}. Skipping."
            )
        loaded_keys.add("context_proj.weight")
    if "context_proj.bias" in pretrained and "context_proj.bias" in model_state:
        model_state["context_proj.bias"].copy_(pretrained["context_proj.bias"])
        loaded_keys.add("context_proj.bias")

    # Blocks: transfer as many as downstream has
    def _count_blocks(state_dict: dict) -> int:
        indices = set()
        for k in state_dict:
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
                indices.add(int(parts[1]))
        return max(indices) + 1 if indices else 0

    num_down_blocks = _count_blocks(model_state)
    num_pre_blocks = _count_blocks(pretrained)
    num_to_transfer = min(num_down_blocks, num_pre_blocks)
    for i in range(num_to_transfer):
        for key in list(model_state.keys()):
            if key.startswith(f"blocks.{i}."):
                src_key = key  # same name since both use "blocks.N."
                if src_key in pretrained:
                    model_state[key].copy_(pretrained[src_key])
                    loaded_keys.add(src_key)

    # Position encoding (SinusoidalPositionalEncoding) — not state_dict params, skip

    loaded_count = len(loaded_keys)
    total_pretrained = len(pretrained)
    print(
        f"[transfer_pretrained_weights] Loaded {loaded_count}/{total_pretrained} keys "
        f"from {Path(pretrained_path).name} (epoch {checkpoint.get('epoch', '?')}, "
        f"val_total_loss={checkpoint.get('metrics', {}).get('val_total_loss', '?'):.4f})"
    )


def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    freeze_backbone_epochs: int = 0,
) -> int:
    """Initialize a model_b (or baseline) from a previous downstream checkpoint.

    Unlike `transfer_pretrained_weights`, this assumes the checkpoint was saved
    by the SAME model class (so state_dict keys match exactly), and only loads
    the model weights — optimizer/scheduler/epoch are NOT restored.

    Use this for curriculum-style fine-tuning: train on a simpler subset (e.g.
    cpg_only mask), then warm-start on a harder dataset (e.g. mask=all) by
    reloading the weights but resetting the optimizer.

    Args:
        model: target model to load weights into.
        checkpoint_path: path to a `.pt` saved by `save_checkpoint` above.
        device: target device.
        freeze_backbone_epochs: if > 0, freeze all parameters except the
            prediction `head` for this many epochs. Set via the optimizer
            after construction (call `unfreeze_backbone()` when done).

    Returns the epoch number from the checkpoint (informational only — the
    outer training loop should start at epoch 1).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained = checkpoint["model_state_dict"]
    model_state = model.state_dict()

    # Load only matching keys; warn about mismatches so silent shape errors
    # don't hide bugs.
    own_keys = set(model_state.keys())
    pre_keys = set(pretrained.keys())
    missing_in_ckpt = own_keys - pre_keys
    unexpected_in_ckpt = pre_keys - own_keys
    if missing_in_ckpt:
        print(f"[load_model_from_checkpoint] WARNING: missing in checkpoint: "
              f"{sorted(missing_in_ckpt)}")
    if unexpected_in_ckpt:
        print(f"[load_model_from_checkpoint] WARNING: unexpected in checkpoint: "
              f"{sorted(unexpected_in_ckpt)}")
    if missing_in_ckpt or unexpected_in_ckpt:
        # Strict by default — names that don't match almost always indicate a
        # model architecture mismatch. Fall back to non-strict load of the
        # intersection so we don't crash before the user can read the warning.
        inter = own_keys & pre_keys
        print(f"[load_model_from_checkpoint] Loading {len(inter)} matching keys "
              f"(non-strict).")
        missing_in_ckpt = sorted(missing_in_ckpt)
        unexpected_in_ckpt = sorted(unexpected_in_ckpt)
        ok_state = {k: pretrained[k] for k in inter}
        ret = model.load_state_dict(ok_state, strict=False)
        if ret.missing_keys or ret.unexpected_keys:
            print(f"[load_model_from_checkpoint] After load_state_dict: "
                  f"missing={ret.missing_keys}, unexpected={ret.unexpected_keys}")
    else:
        model.load_state_dict(pretrained, strict=True)

    ckpt_epoch = int(checkpoint.get("epoch", 0))
    print(
        f"[load_model_from_checkpoint] Loaded weights from "
        f"{Path(checkpoint_path).name} (epoch {ckpt_epoch}, "
        f"val_loss={checkpoint.get('metrics', {}).get('val_loss', float('nan')):.4f})."
    )

    if freeze_backbone_epochs > 0:
        # Freeze everything except the head so the new task's loss doesn't
        # immediately destroy pretrained features.
        for name, p in model.named_parameters():
            if "head" not in name:
                p.requires_grad = False
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[load_model_from_checkpoint] Froze backbone: "
              f"{n_frozen} params frozen, {n_trainable} trainable (head only). "
              f"Will unfreeze after epoch {freeze_backbone_epochs}.")

    return ckpt_epoch


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

    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(mcg_tracks), len(hmcg_tracks), len(atac_tracks))
    seq_len = args.target_length
    post_filter_len = min(seq_len, 4)

    # Fetch real sequences from genome for val metadata (seqs list may be dummy in lazy mode)
    # IMPORTANT: Open and CLOSE genome BEFORE creating DataLoaders.
    # If genome is left open when workers fork(), the inherited file descriptor
    # conflicts with per-worker pyfaidx handles, causing deadlocks.
    genome = pyfaidx.Fasta(args.genome_fasta)

    split_regions_df = df_dmr.iloc[:usable_dmrs].copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
    split_regions_df = assign_non_overlapping_groups(split_regions_df, "chr", "start_expanded", "end_expanded")

    group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * args.train_ratio))
    train_group_ids = set(group_ids[:num_train_groups].tolist())
    train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
    train_indices = np.flatnonzero(train_mask).tolist()
    val_indices = np.flatnonzero(~train_mask).tolist()

    # Write train / val region BED files to output dir for inspection.
    # Output dir is derived from --output-csv's parent directory.
    from pathlib import Path as _Path
    bed_out_dir = _Path(args.output_csv).parent
    write_train_val_beds(
        split_regions_df,
        train_indices,
        val_indices,
        output_dir=bed_out_dir,
        timestamp=args.timestamp,
    )

    # Build val_region_metadata with real sequences fetched from genome
    # (seqs list may be dummy placeholders in lazy mode)
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

    # CLOSE genome before forking workers to prevent file descriptor conflicts
    try:
        genome.close()
    except Exception:
        pass  # pyfaidx might not have a close() method in all versions

    # Store file paths (handles opened per-worker inside the Dataset)
    hm5c_paths = ensure_path_list(getattr(args, "hm5c_bedgraph", None))
    m5c_paths = ensure_path_list(getattr(args, "m5c_bedgraph", None))
    atac_paths = ensure_path_list(getattr(args, "atac_bw", None))

    train_dataset = LazyM5cSequenceAtacDataset(
        indices=train_indices,
        df_dmr=split_regions_df,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=m5c_paths[0],
        hm5c_bedgraph=hm5c_paths[0],
        atac_bw_path=atac_paths[0],
        target_length=args.target_length,
        mask_mode=args.mask_mode,
        atac_scaling=args.atac_scaling,
        augment_rc=getattr(args, "augment_reverse_complement", False),
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
    )

    return PreparedSequenceAtacData(
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
    signal_h5ad: str
    regression_plot: str
    checkpoint_paths: dict


def run_experiment(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> ExperimentResult:
    device = torch.device(f"cuda:{get_freest_gpu()}" if torch.cuda.is_available() else "cpu")
    prepared = prepare_sequence_atac_crosshyena_data(num_dmrs, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)

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
            post_filter_len=prepared.post_filter_len,
            use_positional_encoding=args.use_positional_encoding,
        ).to(device)

    if args.pretrained_checkpoint and args.model_name == "model_b":
        transfer_pretrained_weights(model, args.pretrained_checkpoint, device)

    if getattr(args, "init_from_checkpoint", None):
        # Curriculum warm-start: load weights from a previously-trained
        # downstream checkpoint. Optimizer / scheduler / epoch are NOT
        # restored — fresh training begins from epoch 1.
        ckpt_epoch = load_model_from_checkpoint(
            model,
            args.init_from_checkpoint,
            device,
            freeze_backbone_epochs=getattr(args, "freeze_backbone_epochs", 0),
        )
        print(f"[run_experiment] Warm-started from {Path(args.init_from_checkpoint).name} "
              f"(was at epoch {ckpt_epoch}, now retraining from epoch 1)")

    # Gradient checkpointing: override forward to call torch.checkpoint on each block
    if args.gradient_checkpointing:
        try:
            import torch.utils.checkpoint as ckpt
            orig_forward = model.forward
            def checkpointed_forward(m5c_track, sequence_track, atac_track):
                x = model.query_norm(model.query_proj(m5c_track))
                seq_h = model.sequence_norm(model.sequence_proj(sequence_track))
                atac_h = model.atac_norm(model.atac_proj(atac_track))
                ctx = model.context_norm(model.context_proj(torch.cat([seq_h, atac_h], dim=-1)))
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

    # If we froze the backbone on load, the optimizer was built BEFORE the
    # freeze. Re-create it now so it only sees trainable parameters.
    if getattr(args, "init_from_checkpoint", None) and getattr(args, "freeze_backbone_epochs", 0) > 0:
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args, args.num_epochs)

    # Mixed precision scaler
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
    best_checkpoint_path = args.best_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    last_checkpoint_path = args.last_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    patience_left = args.patience

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch

        # Unfreeze the backbone after the requested freeze epochs (curriculum).
        if (
            getattr(args, "init_from_checkpoint", None)
            and getattr(args, "freeze_backbone_epochs", 0) > 0
            and epoch == args.freeze_backbone_epochs + 1
        ):
            for p in model.parameters():
                p.requires_grad = True
            # Rebuild optimizer so newly-unfrozen params get proper weight_decay.
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(optimizer, args, args.num_epochs)
            n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"[epoch {epoch}] Unfroze backbone: {n_trainable} params trainable.")

        model.train()
        running_loss = 0.0
        seen = 0
        optimizer.zero_grad()
        accum_count = 0

        for m5c_batch, sequence_batch, atac_batch, target_batch, mask_batch in prepared.train_loader:
            m5c_batch = m5c_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                pred = model(m5c_batch, sequence_batch, atac_batch)
                loss = masked_mse_loss(pred, target_batch, mask_batch)
                # Scale loss for gradient accumulation
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
            "signal_h5ad": signal_h5ad,
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
        signal_h5ad=signal_h5ad,
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
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv",
                        help="Path to DMR file. Supports CSV (chr/start/end/length/center columns) or BED (chr, start, end as first three columns).")
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
    parser.add_argument("--sample-sizes", nargs="+", type=str, required=True)
    parser.add_argument("--chromosome", default=None)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--model-name", choices=["baseline", "model_b"], default="baseline")
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
    parser.add_argument("--atac-scaling", choices=["none", "minmax"], default="minmax")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to a MaskedTrackPretrainingModelB checkpoint (.pt) to initialize model_b weights.")
    parser.add_argument("--init-from-checkpoint", default=None, help="Path to a downstream model_b checkpoint (.pt) for warm-start / curriculum fine-tuning. Loads only model weights, resets optimizer/scheduler.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0, help="If >0, freeze all params except head for this many epochs after warm-start from --init-from-checkpoint.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (bfloat16) training. Reduces memory ~2x, recommended for long sequences.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Accumulate gradients over N mini-batches before each optimizer step. Use with --batch-size 4-8 for long sequences.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to trade compute for memory. Recommended for very long sequences (8k+).")
    parser.add_argument("--mask-mode", choices=["cpg_both", "cpg_forward", "c_only", "all"], default="cpg_both")
    parser.add_argument("--augment-reverse-complement", action="store_true")
    parser.add_argument("--use-all-input-groups", action="store_true")
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Enable lazy loading: fetch sequence/track data on-the-fly per batch instead of loading everything into memory upfront.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", default="output/m5c_query_sequence_atac_crosshyena_results.csv")
    parser.add_argument("--output-json", default="output/m5c_query_sequence_atac_crosshyena_results.json")
    parser.add_argument("--timestamp", default="")
    parser.add_argument(
        "--prediction-signal-h5ad",
        default="output/{timestamp}_m5c_query_sequence_atac_crosshyena_prediction_signals_{sample_size}.h5ad",
    )
    parser.add_argument(
        "--regression-plot-path",
        default="output/{timestamp}_m5c_query_sequence_atac_crosshyena_regression_plot_{sample_size}.png",
    )
    parser.add_argument(
        "--best-checkpoint-path",
        default="output/{timestamp}_m5c_query_sequence_atac_crosshyena_best_{sample_size}.pt",
    )
    parser.add_argument(
        "--last-checkpoint-path",
        default="output/{timestamp}_m5c_query_sequence_atac_crosshyena_last_{sample_size}.pt",
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

    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args, lazy=getattr(args, "lazy", False))
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