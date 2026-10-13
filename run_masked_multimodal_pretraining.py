#!/usr/bin/env python
"""Multimodal masked pretraining: shared position mask + per-track modality mask.

Inputs are multiple signal tracks (5mC / 5hmC / ATAC / RNA) plus the DNA
sequence. Output heads default to 5mC + 5hmC (`--reconstruct-tracks`); tracks
that are not reconstructed are input-only (projected into the context, but no
head and never in the loss).

Two masking layers, both implemented with LEARNED [MASK] tokens substituted in
hidden space — never by zeroing the raw values, because 0 is a real methylation
level and zeroing would conflate "missing" with "unmethylated":

  * Position mask (shared across every signal track): each window masks
    `--mask-fraction` of its CpG positions, and at those positions EVERY signal
    track's hidden representation becomes that track's learned [POS-MASK]
    token. The DNA sequence stays visible.
  * Modality mask (configured per track via `--drop-p-{5mc,5hmc,atac,rna,seq}`):
    an absent track's whole hidden representation becomes its learned
    [MODALITY-MASK] token. Sampling follows 方案① — at EVERY training step each
    sample independently re-draws its presence (random per-step + per-sample,
    the dropmod mechanism upgraded from per-batch to per-sample).

Loss is computed ONLY at position-masked positions, so at every predicted
position no signal track is visible as a real value (anti-leakage by
construction); modality masking only removes additional information.

Data pipeline is lazy-only (RNA is only available through the lazy dataset) and
the train/val/test split is the shared three-way split
(`resolve_train_val_test_split`): without `--test-bed` the windows are cut
0.8/0.1/0.1 in genomic order; val drives early stopping, test is evaluated once
after the best checkpoint is selected.

Launch example:

  python run_masked_multimodal_pretraining.py \
    --dmr-csv /data2st1/junyi/generegion_vM23/cCRE_cpg.bed \
    --target-length 16384 --sample-sizes all --num-blocks 2 \
    --m5c-bedgraph  /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz \
    --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz \
    --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw \
    --rna-coverage-bw /data8/junyi/methdata/bulk_rna/BULK_AMY/AMY_MC.bw \
    --reconstruct-tracks 5mc 5hmc --mask-fraction 0.15 \
    --drop-p-atac 0.2 --drop-p-rna 0.2 \
    --scheduler cosine --num-epochs 100 --batch-size 64 --seed 7 \
    --output-csv output/AMY_MC/cCRE_cpg/{timestamp}_masked_mm_results.csv \
    --output-json output/AMY_MC/cCRE_cpg/{timestamp}_masked_mm_results.json \
    --best-checkpoint-path output/AMY_MC/cCRE_cpg/{timestamp}_masked_mm_best_{sample_size}.pt \
    --last-checkpoint-path output/AMY_MC/cCRE_cpg/{timestamp}_masked_mm_last_{sample_size}.pt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import (
    LazyM5cSequenceAtacRnaDataset,
    add_clip_at_zero_argument,
    ensure_path_list,
    generate_pretraining_cpg_mask,
    load_data,
    resolve_train_val_test_split,
    write_train_val_beds,
)
from models import MaskedTrackPretrainingModelB
from utils import get_freest_gpu, resolve_sample_sizes, set_random_seed


ALL_TRACKS = ["5mc", "5hmc", "atac", "rna"]
CONTEXT_ORDER = ["5hmc", "atac", "rna"]          # canonical context order
DEFAULT_RECONSTRUCT_TRACKS = ["5mc", "5hmc"]

# Index of each track inside LazyM5cSequenceAtacRnaDataset.__getitem__'s tuple:
# (m5c, sequence_onehot, atac, hm5c, loss_mask, rna)
_LAZY_TUPLE_INDEX = {"5mc": 0, "atac": 2, "5hmc": 3, "rna": 5}


# ---------------------------------------------------------------------------
# Track layout
# ---------------------------------------------------------------------------

def resolve_track_layout(args) -> tuple[list[str], list[str]]:
    """Return (input_tracks, reconstruct_tracks) in canonical order.

    ``input_tracks[0]`` is the query (always 5mc); context tracks follow with
    the RECONSTRUCTED ones first, so the model's head order
    (``[query] + first N context tracks``) matches ``reconstruct_tracks``.
    """
    reconstruct = list(args.reconstruct_tracks)
    if "5mc" not in reconstruct:
        raise ValueError("--reconstruct-tracks must include '5mc' (the query track).")
    if len(set(reconstruct)) != len(reconstruct):
        raise ValueError("--reconstruct-tracks contains duplicates.")

    if args.input_tracks:
        inputs = list(args.input_tracks)
        unknown = [t for t in inputs if t not in ALL_TRACKS]
        if unknown:
            raise ValueError(f"Unknown --input-tracks entries: {unknown}")
        if "5mc" not in inputs:
            raise ValueError("--input-tracks must include '5mc'.")
        missing = [t for t in reconstruct if t not in inputs]
        if missing:
            raise ValueError(f"--reconstruct-tracks {missing} not present in --input-tracks.")
    else:
        inputs = list(reconstruct)
        if args.atac_bw:
            inputs.append("atac")
        if args.rna_coverage_bw:
            inputs.append("rna")

    recon_ctx = [t for t in reconstruct if t != "5mc"]
    input_only = [t for t in CONTEXT_ORDER if t in inputs and t not in recon_ctx]
    input_tracks = ["5mc"] + recon_ctx + input_only

    if "atac" in input_tracks and not args.atac_bw:
        raise ValueError("ATAC is an input track but --atac-bw was not provided.")
    if "rna" in input_tracks and not args.rna_coverage_bw:
        raise ValueError("RNA is an input track but --rna-coverage-bw was not provided.")
    return input_tracks, reconstruct


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class MaskedMultimodalDataset(torch.utils.data.Dataset):
    """Select the requested input tracks from the lazy dataset's 6-tuple.

    The lazy dataset returns ``(m5c, sequence_onehot, atac, hm5c, loss_mask,
    rna)``. This wrapper returns ``(*tracks, sequence_onehot, base_ids)`` where
    ``tracks`` follows ``input_tracks`` order and ``base_ids`` are recovered
    from the one-hot via argmax (N encodes as index 0 = A, exactly like
    ``build_sequence_tensor``; N never participates in a CpG dinucleotide, so
    mask generation is unaffected). The supervised loss mask is unused here —
    pretraining generates its own shared position mask.
    """

    def __init__(self, base: LazyM5cSequenceAtacRnaDataset, input_tracks: list[str]):
        self.base = base
        self.input_tracks = list(input_tracks)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        m5c, sequence_onehot, atac, hm5c, _loss_mask, rna = self.base[idx]
        source = {"5mc": m5c, "atac": atac, "5hmc": hm5c, "rna": rna}
        tracks = [source[name] for name in self.input_tracks]
        base_ids = sequence_onehot.argmax(dim=-1).long()
        return (*tracks, sequence_onehot, base_ids)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def sample_position_mask(
    base_ids_batch: torch.Tensor,
    mask_fraction: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Shared CpG-position mask (N, L, 1); 1.0 = masked position.

    One mask per sample, applied to EVERY signal track — this is what makes the
    loss anti-leak: at a predicted position no track's real value is visible.
    ``seed=None`` redraws fresh positions every step (training); a fixed seed
    makes validation/test masks reproducible (seed + batch index).
    """
    return generate_pretraining_cpg_mask(base_ids_batch, mask_fraction=mask_fraction, seed=seed)


def sample_track_presence(
    batch_size: int,
    track_names: list[str],
    drop_probs: dict[str, float],
    device: torch.device,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Per-sample modality presence multipliers (N, 1, 1); 1.0 = present.

    方案①: at every step each sample independently re-draws every track's
    presence, so the same region is trained under both "present" and "absent"
    conditions across epochs (no region-identity shortcut). Tracks with p <= 0
    are always present.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    presence = []
    for name in track_names:
        p = float(drop_probs.get(name, 0.0))
        if p <= 0.0:
            presence.append(torch.ones(batch_size, 1, 1, device=device))
        else:
            draw = torch.rand(batch_size, 1, 1, device=device, generator=generator)
            presence.append((draw >= p).float())
    return presence


def sample_sequence_presence(
    batch_size: int,
    drop_p: float,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor | None:
    """Per-sample sequence presence (N, 1, 1); None when sequence is never dropped."""
    if drop_p <= 0.0:
        return None
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    draw = torch.rand(batch_size, 1, 1, device=device, generator=generator)
    return (draw >= drop_p).float()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (pred - target).pow(2)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def compute_masked_reconstruction_loss(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    position_mask: torch.Tensor,
    head_track_names: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {}
    total = 0.0
    for name, pred, target in zip(head_track_names, preds, targets):
        loss = masked_mse_loss(pred, target, position_mask)
        losses[name] = float(loss.detach().cpu().item())
        total = total + loss
    total = total / max(1, len(head_track_names))
    losses["total"] = float(total.detach().cpu().item())
    return total, losses


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: torch.optim.Optimizer, args, steps_per_experiment: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        total_steps = args.scheduler_t_max if args.scheduler_t_max > 0 else steps_per_experiment
        total_steps = max(1, total_steps)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.scheduler_min_lr,
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.scheduler_min_lr,
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


def enable_block_checkpointing(model: nn.Module) -> None:
    """Wrap every StripedHyenaBlock in torch.utils.checkpoint (memory for compute)."""
    import torch.utils.checkpoint as ckpt

    for block in model.blocks:
        original_forward = block.forward

        def wrapped(hidden, context, _orig=original_forward):
            return ckpt.checkpoint(_orig, hidden, context, use_reentrant=False)

        block.forward = wrapped


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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader | None
    usable_dmrs: int
    seq_len: int
    train_regions: int
    val_regions: int
    test_regions: int
    non_overlap_groups: int
    split_info: dict


def prepare_data(num_dmrs: int, args, input_tracks: list[str]) -> PreparedData:
    args.use_m5c = True
    df_dmr, _seqs, _mcg, _hmcg, _atac = load_data(args, lazy=True)
    if len(df_dmr) > num_dmrs:
        df_dmr = df_dmr.iloc[:num_dmrs].copy().reset_index(drop=True)
    usable_dmrs = len(df_dmr)

    split_regions_df = df_dmr.copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)

    train_idx, val_idx, test_idx, split_info = resolve_train_val_test_split(
        split_regions_df,
        train_ratio=args.train_ratio,
        test_bed_path=getattr(args, "test_bed", None),
    )
    write_train_val_beds(
        split_regions_df,
        train_idx,
        val_idx,
        output_dir=Path(args.output_csv).parent,
        timestamp=args.timestamp,
        test_indices=test_idx,
    )

    m5c_path = ensure_path_list(args.m5c_bedgraph)[0]
    hm5c_path = ensure_path_list(args.hm5c_bedgraph)[0]
    atac_paths = ensure_path_list(args.atac_bw)
    atac_path = atac_paths[0] if atac_paths else None
    rna_paths = ensure_path_list(args.rna_coverage_bw)
    rna_path = rna_paths[0] if rna_paths else None

    def _make_loader(indices, shuffle: bool, augment_rc: bool) -> torch.utils.data.DataLoader:
        base = LazyM5cSequenceAtacRnaDataset(
            indices=indices,
            df_dmr=split_regions_df,
            genome_fasta=args.genome_fasta,
            m5c_bedgraph=m5c_path,
            hm5c_bedgraph=hm5c_path,
            atac_bw_path=atac_path,
            target_length=args.target_length,
            mask_mode=args.mask_mode,
            atac_scaling=args.atac_scaling,
            rna_bw_path=rna_path,
            rna_scaling=args.rna_scaling,
            augment_rc=augment_rc,
            clip_at_zero=args.clip_at_zero,
        )
        dataset = MaskedMultimodalDataset(base, input_tracks)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            prefetch_factor=(2 if args.num_workers > 0 else None),
            persistent_workers=False,
            pin_memory=True,
        )

    train_loader = _make_loader(train_idx, shuffle=True, augment_rc=args.augment_reverse_complement)
    val_loader = _make_loader(val_idx, shuffle=False, augment_rc=False)
    test_loader = _make_loader(test_idx, shuffle=False, augment_rc=False) if len(test_idx) > 0 else None

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        usable_dmrs=usable_dmrs,
        seq_len=args.target_length,
        train_regions=len(train_idx),
        val_regions=len(val_idx),
        test_regions=len(test_idx),
        non_overlap_groups=(
            split_info["train_groups"] + split_info["val_groups"] + split_info["test_groups"]
        ),
        split_info=split_info,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    mask_fraction: float,
    head_track_names: list[str],
    reconstruct_indices: list[int],
    amp: bool = False,
    mask_seed: int | None = None,
) -> dict[str, float]:
    """Reconstruction metrics on a held-out loader.

    Position masks are seeded (``mask_seed + batch_index``) so repeated
    evaluations are comparable across epochs; modality dropout is NOT applied
    at eval (dropmod convention) — every track is present, only positions are
    masked. Metrics are computed on masked positions only, per output head.
    """
    model.eval()
    amp_dtype = torch.bfloat16 if amp else torch.float32
    num_input_tracks = model.num_context_tracks + 1
    track_loss_sums = {name: 0.0 for name in head_track_names}
    total_loss_sum = 0.0
    seen = 0
    all_preds = {name: [] for name in head_track_names}
    all_targets = {name: [] for name in head_track_names}
    all_masks = {name: [] for name in head_track_names}

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            tracks = [t.to(device) for t in batch[:num_input_tracks]]
            sequence_batch = batch[num_input_tracks].to(device)
            base_ids_batch = batch[num_input_tracks + 1]

            position_mask = sample_position_mask(
                base_ids_batch,
                mask_fraction,
                seed=None if mask_seed is None else mask_seed + batch_index,
            ).to(device)
            targets = [tracks[i] for i in reconstruct_indices]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
                preds = model(tracks, sequence_batch, position_mask=position_mask)
                total_loss, losses = compute_masked_reconstruction_loss(
                    preds, targets, position_mask, head_track_names
                )

            batch_size = tracks[0].size(0)
            seen += batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            for name in head_track_names:
                track_loss_sums[name] += losses[name] * batch_size

            for name, pred, target in zip(head_track_names, preds, targets):
                all_preds[name].append(pred.detach().cpu())
                all_targets[name].append(target.detach().cpu())
                all_masks[name].append(position_mask.detach().cpu())

    denom = max(seen, 1)
    result = {f"val_{name}_loss": track_loss_sums[name] / denom for name in head_track_names}
    result["val_total_loss"] = total_loss_sum / denom

    for name in head_track_names:
        pred_cat = torch.cat(all_preds[name], dim=0)
        target_cat = torch.cat(all_targets[name], dim=0)
        mask_cat = torch.cat(all_masks[name], dim=0)
        mask_bool = mask_cat.bool()
        masked_pred = pred_cat[mask_bool]
        masked_target = target_cat[mask_bool]

        if masked_pred.numel() < 2:
            result[f"val_{name}_r2"] = float("nan")
            result[f"val_{name}_pearsonr"] = float("nan")
            continue

        ss_res = ((masked_target - masked_pred) ** 2).sum()
        target_mean = masked_target.mean()
        ss_tot = ((masked_target - target_mean) ** 2).sum().clamp_min(1e-12)
        result[f"val_{name}_r2"] = float((1.0 - ss_res / ss_tot).item())

        centered_target = masked_target - masked_target.mean()
        centered_pred = masked_pred - masked_pred.mean()
        denom_pearson = (centered_target.pow(2).sum() * centered_pred.pow(2).sum()).sqrt().clamp_min(1e-12)
        result[f"val_{name}_pearsonr"] = float(((centered_target * centered_pred).sum() / denom_pearson).item())

    return result


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    num_dmrs: int
    train_regions: int
    val_regions: int
    test_regions: int
    non_overlap_groups: int
    best_epoch: int
    final_lr: float
    input_tracks: list[str]
    reconstruct_tracks: list[str]
    mask_fraction: float
    modality_drop_probs: dict
    best_val_total_loss: float
    best_val_track_losses: dict[str, float]
    best_val_r2: dict[str, float]
    best_val_pearsonr: dict[str, float]
    final_val_total_loss: float
    final_val_track_losses: dict[str, float]
    final_val_r2: dict[str, float]
    final_val_pearsonr: dict[str, float]
    test_total_loss: float | None
    test_track_losses: dict[str, float] | None
    test_r2: dict[str, float] | None
    test_pearsonr: dict[str, float] | None
    split_info: dict
    checkpoint_paths: dict


def run_experiment(
    num_dmrs: int,
    args,
    input_tracks: list[str],
    head_track_names: list[str],
    reconstruct_indices: list[int],
) -> ExperimentResult:
    device = torch.device(f"cuda:{get_freest_gpu()}" if torch.cuda.is_available() else "cpu")
    prepared = prepare_data(num_dmrs, args, input_tracks)

    drop_probs = {
        "5mc": args.drop_p_5mc,
        "5hmc": args.drop_p_5hmc,
        "atac": args.drop_p_atac,
        "rna": args.drop_p_rna,
    }

    model = MaskedTrackPretrainingModelB(
        seq_len=prepared.seq_len,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=args.use_positional_encoding,
        num_blocks=args.num_blocks,
        fusion_type=args.fusion_type,
        num_context_tracks=len(input_tracks) - 1,
        num_reconstructed_context_tracks=len(head_track_names) - 1,
        use_mask_tokens=True,
    ).to(device)

    if args.gradient_checkpointing:
        enable_block_checkpointing(model)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.num_epochs)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    amp_dtype = torch.bfloat16 if args.amp else torch.float32

    best_epoch = 0
    best_metrics = {
        "val_total_loss": float("inf"),
        **{f"val_{name}_loss": float("inf") for name in head_track_names},
        **{f"val_{name}_r2": float("-inf") for name in head_track_names},
        **{f"val_{name}_pearsonr": float("-inf") for name in head_track_names},
    }
    best_state = None
    last_epoch = 0
    patience_left = args.patience

    best_checkpoint_path = args.best_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    last_checkpoint_path = args.last_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        model.train()

        train_track_loss_sums = {name: 0.0 for name in head_track_names}
        train_total_loss_sum = 0.0
        seen = 0
        optimizer.zero_grad()
        accum_count = 0

        for batch in prepared.train_loader:
            tracks = [t.to(device) for t in batch[: len(input_tracks)]]
            sequence_batch = batch[len(input_tracks)].to(device)
            base_ids_batch = batch[len(input_tracks) + 1]

            # Layer 1: shared position mask — redrawn every step, applied to all
            # signal tracks. Layer 2: modality mask — 方案①: every step, every
            # sample independently re-draws each track's presence.
            position_mask = sample_position_mask(base_ids_batch, args.mask_fraction).to(device)
            presence = sample_track_presence(
                batch_size=tracks[0].size(0),
                track_names=input_tracks,
                drop_probs=drop_probs,
                device=device,
            )
            sequence_present = sample_sequence_presence(
                batch_size=tracks[0].size(0), drop_p=args.drop_p_seq, device=device,
            )
            targets = [tracks[i] for i in reconstruct_indices]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                preds = model(
                    tracks,
                    sequence_batch,
                    position_mask=position_mask,
                    track_present=presence,
                    sequence_present=sequence_present,
                )
                total_loss, losses = compute_masked_reconstruction_loss(
                    preds, targets, position_mask, head_track_names
                )
                total_loss = total_loss / args.gradient_accumulation_steps

            scaler.scale(total_loss).backward()
            accum_count += 1

            batch_size = tracks[0].size(0)
            seen += batch_size
            train_total_loss_sum += float(total_loss.item()) * batch_size * args.gradient_accumulation_steps
            for name in head_track_names:
                train_track_loss_sums[name] += losses[name] * batch_size

            if accum_count % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        denom = max(seen, 1)
        train_total_loss = train_total_loss_sum / denom
        train_track_losses = {name: train_track_loss_sums[name] / denom for name in head_track_names}

        val_metrics = evaluate(
            model, prepared.val_loader, device, args.mask_fraction,
            head_track_names, reconstruct_indices, amp=args.amp, mask_seed=args.seed + 1000,
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["val_total_loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        train_detail = ", ".join(f"{n}={train_track_losses[n]:.4f}" for n in head_track_names)
        val_detail = ", ".join(
            f"{n}: R²={val_metrics[f'val_{n}_r2']:.4f} ρ={val_metrics[f'val_{n}_pearsonr']:.4f}"
            for n in head_track_names
        )
        print(
            f"[num_dmrs={prepared.usable_dmrs}] Epoch {epoch:02d} | "
            f"train_total={train_total_loss:.4f} ({train_detail}) | "
            f"val_total={val_metrics['val_total_loss']:.4f} | {val_detail} | lr={current_lr:.6g}"
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
                    **{f"train_{n}_loss": train_track_losses[n] for n in head_track_names},
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

    final_val_metrics = evaluate(
        model, prepared.val_loader, device, args.mask_fraction,
        head_track_names, reconstruct_indices, amp=args.amp, mask_seed=args.seed + 1000,
    )
    save_checkpoint(
        last_checkpoint_path,
        model,
        optimizer,
        scheduler,
        last_epoch,
        {**final_val_metrics, "is_best": last_epoch == best_epoch},
        args,
        prepared.usable_dmrs,
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        final_val_metrics = evaluate(
            model, prepared.val_loader, device, args.mask_fraction,
            head_track_names, reconstruct_indices, amp=args.amp, mask_seed=args.seed + 1000,
        )

    # Held-out test: evaluated exactly once on the best checkpoint with its own
    # fixed mask seed, so the reported numbers are reproducible.
    test_metrics: dict[str, float] | None = None
    if prepared.test_loader is not None:
        test_metrics = evaluate(
            model, prepared.test_loader, device, args.mask_fraction,
            head_track_names, reconstruct_indices, amp=args.amp, mask_seed=args.seed + 2000,
        )
        test_detail = ", ".join(
            f"{n}: R²={test_metrics[f'val_{n}_r2']:.4f} ρ={test_metrics[f'val_{n}_pearsonr']:.4f}"
            for n in head_track_names
        )
        print(f"FINAL TEST METRICS (report once): total={test_metrics['val_total_loss']:.4f} | {test_detail}")

    def _track_dict(metrics: dict[str, float], key: str) -> dict[str, float]:
        return {name: float(metrics.get(f"val_{name}_{key}", float("nan"))) for name in head_track_names}

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        train_regions=prepared.train_regions,
        val_regions=prepared.val_regions,
        test_regions=prepared.test_regions,
        non_overlap_groups=prepared.non_overlap_groups,
        best_epoch=best_epoch,
        final_lr=optimizer.param_groups[0]["lr"],
        input_tracks=input_tracks,
        reconstruct_tracks=head_track_names,
        mask_fraction=args.mask_fraction,
        modality_drop_probs=drop_probs,
        best_val_total_loss=best_metrics["val_total_loss"],
        best_val_track_losses=_track_dict(best_metrics, "loss"),
        best_val_r2=_track_dict(best_metrics, "r2"),
        best_val_pearsonr=_track_dict(best_metrics, "pearsonr"),
        final_val_total_loss=final_val_metrics["val_total_loss"],
        final_val_track_losses=_track_dict(final_val_metrics, "loss"),
        final_val_r2=_track_dict(final_val_metrics, "r2"),
        final_val_pearsonr=_track_dict(final_val_metrics, "pearsonr"),
        test_total_loss=(test_metrics or {}).get("val_total_loss"),
        test_track_losses=_track_dict(test_metrics, "loss") if test_metrics is not None else None,
        test_r2=_track_dict(test_metrics, "r2") if test_metrics is not None else None,
        test_pearsonr=_track_dict(test_metrics, "pearsonr") if test_metrics is not None else None,
        split_info=prepared.split_info,
        checkpoint_paths={"best": best_checkpoint_path, "last": last_checkpoint_path},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multimodal masked pretraining: 5mC/5hmC/ATAC/RNA + sequence inputs, "
            "shared position mask + per-track modality mask (learned [MASK] tokens)."
        )
    )
    # data
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv",
                        help="Window set. Supports CSV or BED (chr/start/end).")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--m5c-bedgraph", nargs="+", required=True,
                        help="5mC track(s). With --use-all-input-groups pass one per group.")
    parser.add_argument("--hm5c-bedgraph", nargs="+", required=True,
                        help="5hmC track(s).")
    parser.add_argument("--atac-bw", nargs="+", default=None,
                        help="Optional ATAC bigWig(s); providing them adds ATAC as an input modality.")
    parser.add_argument("--rna-coverage-bw", nargs="+", default=None,
                        help="Optional RNA-coverage bigWig(s); providing them adds RNA as an input modality.")
    parser.add_argument("--use-all-input-groups", action="store_true",
                        help="Pooled multi-group training: every window reads its own group's tracks.")
    # tracks / masking
    parser.add_argument("--input-tracks", nargs="+", choices=ALL_TRACKS, default=None,
                        help="Explicit input modality list (must include 5mc). "
                             "Default: reconstruct-tracks + atac/rna when their paths are provided.")
    parser.add_argument("--reconstruct-tracks", nargs="+", choices=ALL_TRACKS,
                        default=DEFAULT_RECONSTRUCT_TRACKS,
                        help="Tracks that get output heads and enter the loss. "
                             "First entry is the query. Default: 5mc 5hmc.")
    parser.add_argument("--mask-fraction", type=float, default=0.15,
                        help="Fraction of CpG positions masked per window (shared across all tracks).")
    parser.add_argument("--drop-p-5mc", type=float, default=0.0)
    parser.add_argument("--drop-p-5hmc", type=float, default=0.0)
    parser.add_argument("--drop-p-atac", type=float, default=0.0)
    parser.add_argument("--drop-p-rna", type=float, default=0.0)
    parser.add_argument("--drop-p-seq", type=float, default=0.0)
    # geometry / split
    parser.add_argument("--sample-sizes", nargs="+", type=str, required=True)
    parser.add_argument("--target-length", type=int, default=16384)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--test-bed", default=None,
        help="Optional BED of held-out test regions (chr/start/end). Windows overlapping it become the "
             "test set; the rest is split train/val by --train-ratio. Without it, windows are auto-split "
             "train/val/test = train_ratio / (1-train_ratio)/2 / (1-train_ratio)/2 (default 0.8/0.1/0.1).",
    )
    # model
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--fusion-type", choices=["cross_hyena", "cross_attention"], default="cross_hyena")
    parser.add_argument("--use-positional-encoding", action="store_true")
    # training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--augment-reverse-complement", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    # track scaling
    parser.add_argument("--atac-scaling", choices=["none", "minmax"], default="minmax")
    parser.add_argument("--rna-scaling", choices=["none", "minmax"], default="minmax")
    add_clip_at_zero_argument(parser)
    parser.add_argument("--mask-mode", choices=["cpg_both", "cpg_forward", "ch_only", "c_only", "all"],
                        default="cpg_forward",
                        help="Only used to build the lazy dataset's (unused) supervised loss mask.")
    # outputs
    parser.add_argument("--output-csv", default="output/masked_multimodal_pretraining_results.csv")
    parser.add_argument("--output-json", default="output/masked_multimodal_pretraining_results.json")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--best-checkpoint-path", default="output/masked_mm_best_{sample_size}.pt")
    parser.add_argument("--last-checkpoint-path", default="output/masked_mm_last_{sample_size}.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.timestamp:
        args.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    set_random_seed(args.seed)
    args.sample_sizes = resolve_sample_sizes(args.sample_sizes, args)

    input_tracks, reconstruct_tracks = resolve_track_layout(args)
    reconstruct_indices = [input_tracks.index(t) for t in reconstruct_tracks]

    print("=" * 60)
    print("[masked multimodal pretraining]")
    print(f"  input tracks      : {input_tracks}  (query={input_tracks[0]})")
    print(f"  output heads      : {reconstruct_tracks}")
    print(f"  mask fraction     : {args.mask_fraction}  (shared across all signal tracks)")
    print(f"  modality drop p   : 5mc={args.drop_p_5mc} 5hmc={args.drop_p_5hmc} "
          f"atac={args.drop_p_atac} rna={args.drop_p_rna} seq={args.drop_p_seq}")
    print("=" * 60)

    results = []
    for sample_size in args.sample_sizes:
        results.append(
            asdict(run_experiment(sample_size, args, input_tracks, reconstruct_tracks, reconstruct_indices))
        )

    results_df = pd.DataFrame(results)
    print(results_df)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as file_obj:
        json.dump({"args": vars(args), "results": results}, file_obj, indent=2)


if __name__ == "__main__":
    main()
