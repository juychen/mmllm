#!/usr/bin/env python
"""Multi-query masked pretraining: N independent query streams, one shared context.

Each query track (by default 5mC AND 5hmC) is projected and run through the
shared StripedHyena blocks as its own stream; the context (DNA sequence + the
remaining signal tracks, e.g. ATAC/RNA) is fused once and shared. Every query
has its own output head and reconstructs itself.

Two masking layers, both learned-[MASK]-token substitution in hidden space
(never raw zeroing — 0 is a real methylation level):

  * Position mask (shared across every signal track): each window masks
    `--mask-fraction` of its CpG positions; at those positions EVERY signal
    track's hidden — including every query — becomes that track's [POS-MASK]
    token. The DNA sequence stays visible. Loss is computed ONLY at masked
    positions, so at every predicted position no signal track is visible as a
    real value (anti-leak by construction). Query streams never read each other.
  * Modality mask (per track via `--drop-p-{5mc,5hmc,atac,rna,seq}`): an absent
    track's whole hidden becomes its [MODALITY-MASK] token. Sampling is
    per-step + per-sample.

This differs from `run_masked_multimodal_pretraining.py`: there 5hmC is a
CONTEXT track and all heads read the single 5mC trunk; here 5mC and 5hmC are
BOTH query trunks that share only the (seq + ATAC + RNA) context.

Launch example:

  python run_multiquery_pretraining.py \
    --dmr-csv /data2st1/junyi/generegion_vM23/cCRE_cpg.bed \
    --target-length 16384 --sample-sizes all --num-blocks 2 \
    --query-tracks 5mc 5hmc \
    --m5c-bedgraph  /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz \
    --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz \
    --atac-bw /data1st1/junyi/methdata/atac/AMY_MC_track.bw \
    --rna-coverage-bw /data2st1/junyi/output/sn0615/BULK_AMY/AMY_MC.bw \
    --mask-fraction 0.15 \
    --scheduler cosine --num-epochs 100 --batch-size 64 --gradient-accumulation-steps 4 \
    --amp --seed 7 \
    --output-csv output/AMY_MC/cCRE_cpg/{timestamp}_multiquery_results.csv \
    --output-json output/AMY_MC/cCRE_cpg/{timestamp}_multiquery_results.json \
    --best-checkpoint-path output/AMY_MC/cCRE_cpg/{timestamp}_multiquery_best_{sample_size}.pt \
    --last-checkpoint-path output/AMY_MC/cCRE_cpg/{timestamp}_multiquery_last_{sample_size}.pt
"""
from __future__ import annotations

import argparse
import json
import os
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
from models import MultiQueryMaskedPretrainingModel
from utils import (
    derive_test_h5ad_path,
    export_multiquery_predictions_h5ad,
    get_freest_gpu,
    resolve_sample_sizes,
    set_random_seed,
)


ALL_TRACKS = ["5mc", "5hmc", "atac", "rna"]
CONTEXT_ORDER = ["5mc", "5hmc", "atac", "rna"]   # canonical order for non-query context
DEFAULT_QUERY_TRACKS = ["5mc", "5hmc"]

# Index of each track inside LazyM5cSequenceAtacRnaDataset.__getitem__'s tuple:
# (m5c, sequence_onehot, atac, hm5c, loss_mask, rna)
_LAZY_TUPLE_INDEX = {"5mc": 0, "atac": 2, "5hmc": 3, "rna": 5}


# ---------------------------------------------------------------------------
# Track layout
# ---------------------------------------------------------------------------

def resolve_track_layout(args) -> tuple[list[str], list[str]]:
    """Return (query_tracks, context_tracks).

    ``query_tracks`` get their own stream + head; ``context_tracks`` are fused
    (with the DNA sequence) into the shared context. The DNA sequence is always
    context and is not listed here.
    """
    query_tracks = list(args.query_tracks)
    if not query_tracks:
        raise ValueError("--query-tracks must be non-empty.")
    unknown = [t for t in query_tracks if t not in ALL_TRACKS]
    if unknown:
        raise ValueError(f"Unknown --query-tracks entries: {unknown}")
    if len(set(query_tracks)) != len(query_tracks):
        raise ValueError("--query-tracks contains duplicates.")

    available = {"5mc", "5hmc"}
    if args.atac_bw:
        available.add("atac")
    if args.rna_coverage_bw:
        available.add("rna")

    missing = [t for t in query_tracks if t not in available]
    if missing:
        raise ValueError(
            f"--query-tracks {missing} requested but no input path was provided "
            f"(5mc requires --m5c-bedgraph, 5hmc requires --hm5c-bedgraph, "
            f"atac requires --atac-bw, rna requires --rna-coverage-bw)."
        )

    context_tracks = [t for t in CONTEXT_ORDER if t in available and t not in query_tracks]
    return query_tracks, context_tracks


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class MultiQueryDataset(torch.utils.data.Dataset):
    """Select the requested tracks from the lazy dataset's 6-tuple.

    Returns ``(*tracks, sequence_onehot, base_ids)`` where ``tracks`` follows
    ``query_tracks + context_tracks`` order. The lazy dataset returns
    ``(m5c, sequence_onehot, atac, hm5c, loss_mask, rna)``; ``base_ids`` are
    recovered from the one-hot via argmax (N encodes as index 0 = A, exactly like
    ``build_sequence_tensor``; N never participates in a CpG dinucleotide, so
    mask generation is unaffected). The supervised loss mask is unused here —
    pretraining generates its own shared position mask.
    """

    def __init__(self, base: LazyM5cSequenceAtacRnaDataset, track_order: list[str]):
        self.base = base
        self.track_order = list(track_order)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        m5c, sequence_onehot, atac, hm5c, _loss_mask, rna = self.base[idx]
        source = {"5mc": m5c, "atac": atac, "5hmc": hm5c, "rna": rna}
        tracks = [source[name] for name in self.track_order]
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

    One mask per sample, applied to EVERY signal track (including every query) —
    this is what makes the loss anti-leak: at a predicted position no track's
    real value is visible. ``seed=None`` redraws fresh positions every step
    (training); a fixed seed makes validation/test masks reproducible.
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

    At every step each sample independently re-draws every track's presence, so
    the same region is trained under both "present" and "absent" conditions
    across epochs (no region-identity shortcut). Tracks with p <= 0 stay present.
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
    """Per-sample sequence presence (N, 1, 1); None when the sequence never drops."""
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
    """Wrap every StripedHyenaBlock in torch.utils.checkpoint (memory for compute).

    Query-stream agnostic: the blocks are shared, so each block is checkpointed
    once per query-stream pass per step automatically.
    """
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


def prepare_data(num_dmrs: int, args, track_order: list[str]) -> PreparedData:
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
        dataset = MultiQueryDataset(base, track_order)
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

def _loader_region_metadata(loader, n_samples: int) -> "pd.DataFrame":
    """Region metadata (chr/start/end/...) in the order ``loader`` yields samples.

    Only valid for a sequential, non-augmented loader (val/test): the exported
    rows must align 1:1 with the prediction tensor, so shuffling and
    reverse-complement doubling are rejected rather than silently misaligned.
    """
    dataset = loader.dataset
    base = getattr(dataset, "base", dataset)
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
        raise ValueError("prediction export requires a sequential loader (shuffle=False)")
    if getattr(base, "augment_rc", False):
        raise ValueError("prediction export requires augment_rc=False")
    if n_samples != len(base):
        raise ValueError(f"loader yielded {n_samples} samples but dataset has {len(base)}")
    rows = []
    for k in range(n_samples):
        row = base.df_dmr.iloc[base.indices[k % base.N]]
        entry = {
            "chr": str(row["chr"]),
            "start_expanded": int(row["start_expanded"]),
            "end_expanded": int(row["end_expanded"]),
        }
        if "strand_view" in row.index:
            entry["strand_view"] = row["strand_view"]
        if "original_idx" in row.index:
            entry["original_idx"] = int(row["original_idx"])
        rows.append(entry)
    return pd.DataFrame(rows)


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    mask_fraction: float,
    query_tracks: list[str],
    context_tracks: list[str],
    amp: bool = False,
    mask_seed: int | None = None,
    return_predictions: bool = False,
):
    """Reconstruction metrics on a held-out loader (query tracks are the heads).

    Position masks are seeded (``mask_seed + batch_index``) so repeated
    evaluations are comparable across epochs; modality dropout is NOT applied at
    eval — every track is present, only positions are masked. Metrics are
    computed on masked positions only, per query head.

    With ``return_predictions=True`` returns ``(metrics, payload)`` where payload
    carries dense per-window ``predictions``/``targets``/``mask`` arrays,
    ``base_ids`` and the matching ``region_metadata`` — ready to hand to
    :func:`utils.export_multiquery_predictions_h5ad`.
    """
    model.eval()
    amp_dtype = torch.bfloat16 if amp else torch.float32
    num_q = len(query_tracks)
    num_c = len(context_tracks)
    track_loss_sums = {name: 0.0 for name in query_tracks}
    total_loss_sum = 0.0
    seen = 0
    all_preds = {name: [] for name in query_tracks}
    all_targets = {name: [] for name in query_tracks}
    all_masks = {name: [] for name in query_tracks}
    all_base_ids: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            queries = [t.to(device) for t in batch[:num_q]]
            context = [t.to(device) for t in batch[num_q:num_q + num_c]]
            sequence_batch = batch[num_q + num_c].to(device)
            base_ids_batch = batch[num_q + num_c + 1]

            position_mask = sample_position_mask(
                base_ids_batch,
                mask_fraction,
                seed=None if mask_seed is None else mask_seed + batch_index,
            ).to(device)
            targets = queries

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
                preds = model(queries, sequence_batch, context, position_mask=position_mask)
                total_loss, losses = compute_masked_reconstruction_loss(
                    preds, targets, position_mask, query_tracks
                )

            batch_size = queries[0].size(0)
            seen += batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            for name in query_tracks:
                track_loss_sums[name] += losses[name] * batch_size

            for name, pred, target in zip(query_tracks, preds, targets):
                all_preds[name].append(pred.detach().cpu())
                all_targets[name].append(target.detach().cpu())
                all_masks[name].append(position_mask.detach().cpu())
            if return_predictions:
                all_base_ids.append(base_ids_batch.detach().cpu())

    denom = max(seen, 1)
    result = {f"val_{name}_loss": track_loss_sums[name] / denom for name in query_tracks}
    result["val_total_loss"] = total_loss_sum / denom

    for name in query_tracks:
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

    if not return_predictions:
        return result

    payload = {
        "predictions": {n: torch.cat(all_preds[n], dim=0).squeeze(-1).float().numpy() for n in query_tracks},
        "targets": {n: torch.cat(all_targets[n], dim=0).squeeze(-1).float().numpy() for n in query_tracks},
        # one shared position mask by construction — any track's copy is the same
        "mask": torch.cat(all_masks[query_tracks[0]], dim=0).squeeze(-1).float().numpy(),
        "base_ids": torch.cat(all_base_ids, dim=0).numpy().astype(np.uint8),
        "region_metadata": _loader_region_metadata(loader, seen),
    }
    return result, payload


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
    query_tracks: list[str]
    context_tracks: list[str]
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
    test_h5ad_path: str = ""


def run_experiment(
    num_dmrs: int,
    args,
    query_tracks: list[str],
    context_tracks: list[str],
) -> ExperimentResult:
    if torch.cuda.is_available():
        # Respect an explicit CUDA_VISIBLE_DEVICES pin (only that device is
        # visible → it is cuda:0); otherwise pick the freest GPU.
        device = torch.device("cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else f"cuda:{get_freest_gpu()}")
    else:
        device = torch.device("cpu")
    track_order = query_tracks + context_tracks
    prepared = prepare_data(num_dmrs, args, track_order)

    drop_probs = {
        "5mc": args.drop_p_5mc,
        "5hmc": args.drop_p_5hmc,
        "atac": args.drop_p_atac,
        "rna": args.drop_p_rna,
    }

    model = MultiQueryMaskedPretrainingModel(
        seq_len=prepared.seq_len,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=args.use_positional_encoding,
        num_blocks=args.num_blocks,
        fusion_type=args.fusion_type,
        num_queries=len(query_tracks),
        num_context_tracks=len(context_tracks),
        context_mode=args.context_mode,
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
        **{f"val_{name}_loss": float("inf") for name in query_tracks},
        **{f"val_{name}_r2": float("-inf") for name in query_tracks},
        **{f"val_{name}_pearsonr": float("-inf") for name in query_tracks},
    }
    best_state = None
    last_epoch = 0
    patience_left = args.patience

    best_checkpoint_path = args.best_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)
    last_checkpoint_path = args.last_checkpoint_path.format(sample_size=prepared.usable_dmrs, timestamp=args.timestamp)

    num_q = len(query_tracks)
    num_c = len(context_tracks)

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        model.train()

        train_track_loss_sums = {name: 0.0 for name in query_tracks}
        train_total_loss_sum = 0.0
        seen = 0
        optimizer.zero_grad()
        accum_count = 0

        for batch in prepared.train_loader:
            queries = [t.to(device) for t in batch[:num_q]]
            context = [t.to(device) for t in batch[num_q:num_q + num_c]]
            sequence_batch = batch[num_q + num_c].to(device)
            base_ids_batch = batch[num_q + num_c + 1]

            # Layer 1: shared position mask (all signal tracks, incl. every query).
            # Layer 2: modality mask — each track's presence re-drawn per step/sample.
            position_mask = sample_position_mask(base_ids_batch, args.mask_fraction).to(device)
            query_present = sample_track_presence(
                batch_size=queries[0].size(0),
                track_names=query_tracks,
                drop_probs=drop_probs,
                device=device,
            )
            track_present = sample_track_presence(
                batch_size=queries[0].size(0),
                track_names=context_tracks,
                drop_probs=drop_probs,
                device=device,
            )
            sequence_present = sample_sequence_presence(
                batch_size=queries[0].size(0), drop_p=args.drop_p_seq, device=device,
            )
            targets = queries

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                preds = model(
                    queries,
                    sequence_batch,
                    context,
                    position_mask=position_mask,
                    query_present=query_present,
                    track_present=track_present,
                    sequence_present=sequence_present,
                )
                total_loss, losses = compute_masked_reconstruction_loss(
                    preds, targets, position_mask, query_tracks
                )
                total_loss = total_loss / args.gradient_accumulation_steps

            scaler.scale(total_loss).backward()
            accum_count += 1

            batch_size = queries[0].size(0)
            seen += batch_size
            train_total_loss_sum += float(total_loss.item()) * batch_size * args.gradient_accumulation_steps
            for name in query_tracks:
                train_track_loss_sums[name] += losses[name] * batch_size

            if accum_count % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        denom = max(seen, 1)
        train_total_loss = train_total_loss_sum / denom
        train_track_losses = {name: train_track_loss_sums[name] / denom for name in query_tracks}

        val_metrics = evaluate(
            model, prepared.val_loader, device, args.mask_fraction,
            query_tracks, context_tracks, amp=args.amp, mask_seed=args.seed + 1000,
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["val_total_loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        train_detail = ", ".join(f"{n}={train_track_losses[n]:.4f}" for n in query_tracks)
        val_detail = ", ".join(
            f"{n}: R²={val_metrics[f'val_{n}_r2']:.4f} ρ={val_metrics[f'val_{n}_pearsonr']:.4f}"
            for n in query_tracks
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
                    **{f"train_{n}_loss": train_track_losses[n] for n in query_tracks},
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
        query_tracks, context_tracks, amp=args.amp, mask_seed=args.seed + 1000,
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
            query_tracks, context_tracks, amp=args.amp, mask_seed=args.seed + 1000,
        )

    # Held-out test: evaluated exactly once on the best checkpoint with its own
    # fixed mask seed, so the reported numbers are reproducible.  The per-window
    # predictions are exported to .h5ad (on by default) for downstream analysis.
    test_metrics: dict[str, float] | None = None
    test_h5ad_path = ""
    if prepared.test_loader is not None:
        test_payload = None
        if args.export_test_h5ad:
            test_metrics, test_payload = evaluate(
                model, prepared.test_loader, device, args.mask_fraction,
                query_tracks, context_tracks, amp=args.amp, mask_seed=args.seed + 2000,
                return_predictions=True,
            )
        else:
            test_metrics = evaluate(
                model, prepared.test_loader, device, args.mask_fraction,
                query_tracks, context_tracks, amp=args.amp, mask_seed=args.seed + 2000,
            )
        test_detail = ", ".join(
            f"{n}: R²={test_metrics[f'val_{n}_r2']:.4f} ρ={test_metrics[f'val_{n}_pearsonr']:.4f}"
            for n in query_tracks
        )
        print(f"FINAL TEST METRICS (report once): total={test_metrics['val_total_loss']:.4f} | {test_detail}")

        if test_payload is not None:
            h5ad_template = args.test_h5ad_path or derive_test_h5ad_path(best_checkpoint_path)
            test_h5ad_path = h5ad_template.format(
                sample_size=prepared.usable_dmrs, timestamp=args.timestamp,
            )
            try:
                export_multiquery_predictions_h5ad(
                    test_h5ad_path,
                    test_payload["region_metadata"],
                    test_payload["base_ids"],
                    test_payload["predictions"],
                    test_payload["targets"],
                    test_payload["mask"],
                    track_names=query_tracks,
                    positions=args.test_h5ad_positions,
                    uns_extra={
                        "split": "test",
                        "mask_fraction": args.mask_fraction,
                        "test_r2": {n: test_metrics[f"val_{n}_r2"] for n in query_tracks},
                        "test_pearsonr": {n: test_metrics[f"val_{n}_pearsonr"] for n in query_tracks},
                        "checkpoint": str(best_checkpoint_path),
                    },
                )
            except Exception as exc:                     # never fail a finished run on export
                test_h5ad_path = ""
                print(f"[h5ad] WARNING: test prediction export failed: {exc}")

    def _track_dict(metrics: dict[str, float], key: str) -> dict[str, float]:
        return {name: float(metrics.get(f"val_{name}_{key}", float("nan"))) for name in query_tracks}

    return ExperimentResult(
        num_dmrs=prepared.usable_dmrs,
        train_regions=prepared.train_regions,
        val_regions=prepared.val_regions,
        test_regions=prepared.test_regions,
        non_overlap_groups=prepared.non_overlap_groups,
        best_epoch=best_epoch,
        final_lr=optimizer.param_groups[0]["lr"],
        query_tracks=query_tracks,
        context_tracks=context_tracks,
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
        test_h5ad_path=test_h5ad_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-query masked pretraining: N independent query streams (e.g. 5mC and 5hmC), "
            "each with its own head, sharing one context (sequence + remaining tracks). "
            "Shared position mask + per-track modality mask (learned [MASK] tokens)."
        )
    )
    # data
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv",
                        help="Window set. Supports CSV or BED (chr/start/end).")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--m5c-bedgraph", nargs="+", required=True,
                        help="5mC track(s). Required (5mc is a default query track).")
    parser.add_argument("--hm5c-bedgraph", nargs="+", required=True,
                        help="5hmC track(s). Required (5hmc is a default query track).")
    parser.add_argument("--atac-bw", nargs="+", default=None,
                        help="Optional ATAC bigWig(s); providing them adds ATAC as a context track.")
    parser.add_argument("--rna-coverage-bw", nargs="+", default=None,
                        help="Optional RNA-coverage bigWig(s); providing them adds RNA as a context track.")
    parser.add_argument("--use-all-input-groups", action="store_true",
                        help="Pooled multi-group training: every window reads its own group's tracks.")
    # tracks / masking
    parser.add_argument("--query-tracks", nargs="+", choices=ALL_TRACKS,
                        default=DEFAULT_QUERY_TRACKS,
                        help="Tracks that get their own query stream + output head and enter the loss. "
                             "All other available input tracks become context (with the DNA sequence). "
                             "Default: 5mc 5hmc.")
    parser.add_argument("--context-mode", choices=["shared", "all-others"], default="shared",
                        help="What each query's context carries. "
                             "'shared' (default): DNA sequence + the non-query tracks, identical for every "
                             "query (queries do NOT see each other). "
                             "'all-others': DNA sequence + every other signal track, so each query also sees "
                             "the other query tracks (e.g. 5hmC keeps access to 5mC) via its own context.")
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
    parser.add_argument("--output-csv", default="output/multiquery_pretraining_results.csv")
    parser.add_argument("--output-json", default="output/multiquery_pretraining_results.json")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--best-checkpoint-path", default="output/multiquery_mm_best_{sample_size}.pt")
    parser.add_argument("--last-checkpoint-path", default="output/multiquery_mm_last_{sample_size}.pt")
    parser.add_argument("--export-test-h5ad", action=argparse.BooleanOptionalAction, default=True,
                        help="Export the held-out TEST predictions/targets/mask to .h5ad after the final "
                             "test evaluation (default: on; disable with --no-export-test-h5ad).")
    parser.add_argument("--test-h5ad-path",
                        default="",
                        help="Output template for the test .h5ad ({sample_size} and {timestamp} are "
                             "substituted). Empty (default) derives it from --best-checkpoint-path by "
                             "replacing '_best_' with '_test_pred_' and '.pt' with '.h5ad'.")
    parser.add_argument("--test-h5ad-positions", choices=["cpg", "cpg_forward", "c"], default="cpg",
                        help="Positions stored in the test .h5ad: 'cpg' (C and G of every CpG; matches the "
                             "masked metric, default), 'cpg_forward' (forward-strand C only) or 'c' (all C).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.timestamp:
        args.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    set_random_seed(args.seed)
    args.sample_sizes = resolve_sample_sizes(args.sample_sizes, args)

    query_tracks, context_tracks = resolve_track_layout(args)

    print("=" * 60)
    print("[multi-query masked pretraining]")
    print(f"  query tracks      : {query_tracks}  (one stream + head each)")
    print(f"  context tracks    : {context_tracks}  (+ DNA sequence)")
    print(f"  context mode      : {args.context_mode}"
          + ("  (each query also sees the OTHER queries)" if args.context_mode == "all-others"
             else "  (queries do NOT see each other)"))
    print(f"  mask fraction     : {args.mask_fraction}  (shared across all signal tracks)")
    print(f"  modality drop p   : 5mc={args.drop_p_5mc} 5hmc={args.drop_p_5hmc} "
          f"atac={args.drop_p_atac} rna={args.drop_p_rna} seq={args.drop_p_seq}")
    print(f"  blocks / hidden   : {args.num_blocks} / {args.hidden_dim}  (blocks shared across queries)")
    print(f"  test pred .h5ad   : {'on' if args.export_test_h5ad else 'off'}"
          + (f"  → {args.test_h5ad_path}" if args.test_h5ad_path else "  (derived from best-checkpoint path)"))
    print("=" * 60)

    results = []
    for sample_size in args.sample_sizes:
        results.append(
            asdict(run_experiment(sample_size, args, query_tracks, context_tracks))
        )

    results_df = pd.DataFrame(results)
    print(results_df)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as file_obj:
        json.dump({"args": vars(args), "results": results}, file_obj, indent=2)


if __name__ == "__main__":
    main()
