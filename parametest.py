#!/usr/bin/env python
"""parametest.py — standalone hyper-parameter screener for the 2-head multiquery model.

Why this file exists
--------------------
`run_multiquery_pretraining.py` is the production trainer: 100 epochs, ~40 min per
epoch on the full cCRE set, early stopping on `val_total_loss`.  Tuning against it
is impossible (one config = days) and its validation mask is drawn per *batch*
(`seed + batch_index`), so changing `--batch-size` also changes which CpGs are
scored -- two configs are then not comparable.

This script is deliberately SEPARATE: it imports only the library layer
(`data.py`, `models.py`, `utils.py`) and implements its own short training loop.
Nothing in the existing pipeline is imported or modified.

Guarantees
----------
1. train / val / test are ALWAYS all three built.  Test is evaluated once per
   config from the best-validation checkpoint, and the three BED files are
   written to disk so the split is auditable and reproducible.
2. Validation masks are PER-REGION deterministic (crc32(chr:start) ^ base_seed).
   A window therefore gets the same masked CpGs no matter the batch size, the
   gradient-accumulation factor, or the loader order.  This is what makes
   lr / batch / hidden-dim comparisons apples-to-apples.
3. Metrics are accumulated in float64 as global sufficient statistics
   (n, Sp, St, Spp, Stt, Spt, Sres), so R^2 and Pearson are exact over ALL masked
   positions, with O(1) memory.
4. Every epoch is appended to a history JSONL, so learning curves are analysable
   without grepping stdout.

Screening budget (important)
----------------------------
A screening run gets ~1 h of GPU, i.e. roughly 30k-50k windows = ~150-200
optimizer steps at effective batch 256.  A 5-epoch COSINE anneal is the wrong
schedule for that budget: it decays `lr` to `min_lr` while most of the steps are
still being taken, so the ranking degenerates into "who moved fastest in the
first 68 steps" and systematically favours a large `lr`.  Screening therefore
forces a CONSTANT lr (`--force-scheduler none`, the default) and compares configs
at matched step count; the real cosine schedule is only used for the final
confirmation runs.  Use `--force-scheduler cosine` to reproduce the production
schedule instead.

Protocol (fixed unless overridden by a config)
----------------------------------------------
query tracks   : 5mc, 5hmc (2 independent streams, 2 heads)
context        : DNA sequence + ATAC + RNA, context_mode="all-others"
loss           : masked MSE at CpG-pair positions, averaged over the 2 heads
selection      : best `val_total_loss` checkpoint -> then scored on test
augmentation   : reverse-complement on TRAIN only (val/test untouched)

Examples
--------
    # smoke test: 480 windows, 1 epoch, one config
    python parametest.py --smoke

    # one shard of the built-in grid (3 GPUs -> shard 0/3, 1/3, 2/3)
    CUDA_VISIBLE_DEVICES=5 python parametest.py --shard 0/3 --epochs 5
    CUDA_VISIBLE_DEVICES=6 python parametest.py --shard 1/3 --epochs 5
    CUDA_VISIBLE_DEVICES=7 python parametest.py --shard 2/3 --epochs 5

    # a single named config, full data, 5 epochs
    python parametest.py --only baseline --proxy-size 0 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import (
    LazyM5cSequenceAtacRnaDataset,
    add_clip_at_zero_argument,
    find_forward_cpg_positions,
    load_data,
    resolve_train_val_test_split,
    write_train_val_beds,
)
from models import MultiQueryMaskedPretrainingModel
from utils import get_freest_gpu, set_random_seed

QUERY_TRACKS = ["5mc", "5hmc"]
CONTEXT_TRACKS = ["atac", "rna"]
TRACK_ORDER = QUERY_TRACKS + CONTEXT_TRACKS

# Index of each track inside LazyM5cSequenceAtacRnaDataset.__getitem__'s tuple:
# (m5c, sequence_onehot, atac, hm5c, loss_mask, rna)
_LAZY_TUPLE_INDEX = {"5mc": 0, "atac": 2, "5hmc": 3, "rna": 5}

DEFAULTS = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "grad_accum": 8,
    "hidden_dim": 64,
    "num_blocks": 2,
    "mask_fraction": 0.15,
    "neighborhood_bp": 50,
    "weight_decay": 1e-5,
    "min_lr": 1e-5,
    "scheduler": "cosine",      # cosine | none
    "warmup_epochs": 0,
    "target_length": 16384,
    "drop_p_atac": 0.1,
    "drop_p_rna": 0.1,
    "seed": 7,
}

# ---------------------------------------------------------------------------
# Built-in screening grid: (name, overrides on DEFAULTS)
# 18 configs, all at 5 epochs on the proxy set.  Ordered so that shard 0 starts
# with the baseline.
# ---------------------------------------------------------------------------
GRID: list[tuple[str, dict]] = [
    ("baseline",      {}),
    # ---- learning rate ------------------------------------------------
    ("lr-6e4",        {"learning_rate": 6e-4}),
    ("lr-2e3",        {"learning_rate": 2e-3, "warmup_epochs": 1}),
    # ---- capacity -----------------------------------------------------
    # NOTE: 16k x 2 streams already uses ~35.6 GB at bs32/h64 (measured), so the
    # wider/deeper variants halve the micro-batch and double accumulation to keep
    # the effective batch at 256 while staying inside 80 GB.
    ("h96",           {"hidden_dim": 96}),
    ("h128",          {"hidden_dim": 128, "batch_size": 16, "grad_accum": 16}),
    ("b3",            {"num_blocks": 3, "batch_size": 16, "grad_accum": 16}),
    # ---- lr x capacity ------------------------------------------------
    ("lr2e3-h96",     {"learning_rate": 2e-3, "warmup_epochs": 1, "hidden_dim": 96}),
    ("lr6e4-h96",     {"learning_rate": 6e-4, "hidden_dim": 96}),
    ("lr2e3-h128",    {"learning_rate": 2e-3, "warmup_epochs": 1, "hidden_dim": 128,
                       "batch_size": 16, "grad_accum": 16}),
    # ---- effective batch (32x8 = 256 baseline) ------------------------
    ("bs16-acc16",    {"batch_size": 16, "grad_accum": 16}),
    ("bs32-acc16",    {"batch_size": 32, "grad_accum": 16}),   # effective 512
    ("bs16-acc8",     {"batch_size": 16, "grad_accum": 8}),    # effective 128
    # ---- masking task difficulty --------------------------------------
    ("mf25",          {"mask_fraction": 0.25}),
    ("bp100",         {"neighborhood_bp": 100}),
    ("mf25-bp100",    {"mask_fraction": 0.25, "neighborhood_bp": 100}),
    # ---- regularisation -------------------------------------------------
    ("wd1e2",         {"weight_decay": 1e-2}),
    # ---- finer lr around the base -------------------------------------
    ("lr5e4",         {"learning_rate": 5e-4}),
    # ---- capacity combo -------------------------------------------------
    ("h96-b3",        {"hidden_dim": 96, "num_blocks": 3, "batch_size": 16, "grad_accum": 16}),
]


# ---------------------------------------------------------------------------
# Deterministic per-region masking
# ---------------------------------------------------------------------------

def region_seed(chrom: str, start: int, mask_seed: int) -> int:
    """Stable 31-bit seed from a window's identity (batch/order independent)."""
    h = zlib.crc32(f"{chrom}:{start}".encode())
    return int((h * 2654435761 + mask_seed) % (2**31 - 1))


def pair_masks(
    base_ids: torch.Tensor,
    seeds: np.ndarray,
    mask_fraction: float,
    neighborhood_bp: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(hide_mask, loss_mask), both (N, L, 1) float32 on CPU.

    Identical semantics to `data.generate_cpg_pair_mask` -- select
    `mask_fraction` of the forward C of each CpG, hide C/G +/- neighborhood_bp,
    score only the C -- but each sample draws from its OWN seed, so a sample's
    mask never depends on which other samples share its batch.
    """
    n, length = base_ids.shape
    bp = max(0, int(neighborhood_bp))
    cpg_c = find_forward_cpg_positions(base_ids)          # (N, L) bool
    loss = torch.zeros((n, length, 1), dtype=torch.float32)
    for i in range(n):
        idx = torch.nonzero(cpg_c[i], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        rng = np.random.RandomState(int(seeds[i]) % (2**31 - 1))
        num = max(1, int(idx.numel() * float(mask_fraction)))
        selected = rng.choice(idx.numpy(), size=num, replace=False)
        loss[i, selected, 0] = 1.0
    # neighborhood buffer: exact, vectorised (max over a 2bp+1 window)
    hide = F.max_pool1d(
        loss.permute(0, 2, 1), kernel_size=2 * bp + 1, stride=1, padding=bp
    ).permute(0, 2, 1).contiguous()
    return hide, loss


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TrackSelector(torch.utils.data.Dataset):
    """Pick the requested tracks out of the lazy dataset's fixed 6-tuple."""

    def __init__(self, base, track_order: list[str]):
        self.base = base
        self.order = list(track_order)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        m5c, seq, atac, hm5c, _loss_mask, rna = self.base[idx]
        src = {"5mc": m5c, "atac": atac, "5hmc": hm5c, "rna": rna}
        tracks = [src[name] for name in self.order]
        base_ids = seq.argmax(dim=-1).long()
        return (*tracks, seq, base_ids)


def build_split(args, cfg, quiet: bool = False):
    """Load metadata, (optionally) subsample, and cut train/val/test."""
    fake = argparse.Namespace(
        dmr_csv=args.dmr_csv,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=[args.m5c_bedgraph],
        hm5c_bedgraph=[args.hm5c_bedgraph],
        atac_bw=[args.atac_bw] if args.atac_bw else None,
        rna_coverage_bw=[args.rna_coverage_bw] if args.rna_coverage_bw else None,
        target_length=cfg["target_length"],
        sample_sizes=None,          # no prefix truncation -- we subsample below
        use_all_input_groups=False,
        use_m5c=True,
        clip_at_zero=False,
    )
    df, *_ = load_data(fake, lazy=True)

    if args.proxy_size and args.proxy_size < len(df):
        rng = np.random.RandomState(args.proxy_seed)
        keep = np.sort(rng.choice(len(df), size=args.proxy_size, replace=False))
        df = df.iloc[keep].copy().reset_index(drop=True)

    frame = df.copy().reset_index().rename(columns={"index": "original_idx"})
    frame["chr"] = frame["chr"].astype(str)
    frame["start_expanded"] = frame["start_expanded"].astype(int)
    frame["end_expanded"] = frame["end_expanded"].astype(int)

    train_idx, val_idx, test_idx, info = resolve_train_val_test_split(
        frame,
        train_ratio=args.train_ratio,
        test_bed_path=args.test_bed if args.test_bed and Path(args.test_bed).exists() else None,
    )
    return frame, train_idx, val_idx, test_idx, info


def make_loader(frame, indices, args, cfg, shuffle: bool, augment_rc: bool):
    base = LazyM5cSequenceAtacRnaDataset(
        indices=list(indices),
        df_dmr=frame,
        genome_fasta=args.genome_fasta,
        m5c_bedgraph=args.m5c_bedgraph,
        hm5c_bedgraph=args.hm5c_bedgraph,
        atac_bw_path=args.atac_bw or None,
        target_length=cfg["target_length"],
        mask_mode="cpg_forward",
        atac_scaling="minmax",
        rna_bw_path=args.rna_coverage_bw or None,
        rna_scaling="minmax",
        augment_rc=augment_rc,
        clip_at_zero=False,
    )
    ds = TrackSelector(base, TRACK_ORDER)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        # deterministic sample order is REQUIRED for per-region val seeds
        worker_init_fn=None,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Evaluation (exact global R^2 / Pearson, O(1) memory)
# ---------------------------------------------------------------------------

class HeadStats:
    def __init__(self) -> None:
        self.n = 0
        self.sp = self.st = self.spp = self.stt = self.spt = self.sres = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool().expand_as(pred)
        if not bool(m.any()):
            return
        p = pred[m].double()
        t = target[m].double()
        self.n += int(p.numel())
        self.sp += float(p.sum())
        self.st += float(t.sum())
        self.spp += float((p * p).sum())
        self.stt += float((t * t).sum())
        self.spt += float((p * t).sum())
        self.sres += float(((p - t) ** 2).sum())

    def metrics(self) -> dict:
        if self.n < 2:
            return {"loss": float("nan"), "r2": float("nan"), "pearsonr": float("nan")}
        n = float(self.n)
        sst = self.stt - self.st * self.st / n
        ssp = self.spp - self.sp * self.sp / n
        cov = self.spt - self.sp * self.st / n
        r2 = 1.0 - self.sres / sst if sst > 0 else float("nan")
        pear = cov / np.sqrt(max(sst, 1e-12) * max(ssp, 1e-12)) if sst > 0 and ssp > 0 else float("nan")
        return {"loss": self.sres / n, "r2": float(r2), "pearsonr": float(pear)}


@torch.no_grad()
def evaluate(model, loader, seeds, cfg, device, amp: bool, desc: str = "val") -> dict:
    model.eval()
    amp_dtype = torch.bfloat16 if amp else torch.float32
    stats = {name: HeadStats() for name in QUERY_TRACKS}
    seen = 0
    nq = len(QUERY_TRACKS)
    nc = len(CONTEXT_TRACKS)

    for batch in loader:
        queries = [t.to(device) for t in batch[:nq]]
        context = [t.to(device) for t in batch[nq:nq + nc]]
        seq = batch[nq + nc].to(device)
        base_ids = batch[nq + nc + 1]

        bsz = base_ids.size(0)
        batch_seeds = seeds[seen:seen + bsz]
        if len(batch_seeds) != bsz:                      # defensive: loader order drifted
            raise RuntimeError(f"{desc}: seed/order mismatch at offset {seen}")
        seen += bsz

        hide, loss_mask = pair_masks(base_ids, batch_seeds, cfg["mask_fraction"], cfg["neighborhood_bp"])
        hide, loss_mask = hide.to(device), loss_mask.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp and device.type == "cuda"):
            preds = model(queries, seq, context, position_mask=hide)

        for name, pred, target in zip(QUERY_TRACKS, preds, queries):
            stats[name].update(pred.detach().cpu(), target.detach().cpu(), loss_mask.detach().cpu())

    out = {"windows": seen}
    total = 0.0
    for name in QUERY_TRACKS:
        m = stats[name].metrics()
        for key, val in m.items():
            out[f"{desc}_{name}_{key}"] = val
        total += m["loss"] if np.isfinite(m["loss"]) else 0.0
    out[f"{desc}_total_loss"] = total / max(1, len(QUERY_TRACKS))
    return out


# ---------------------------------------------------------------------------
# Model / optimiser
# ---------------------------------------------------------------------------

def build_model(cfg, seq_len: int, device) -> nn.Module:
    return MultiQueryMaskedPretrainingModel(
        seq_len=seq_len,
        hidden_dim=cfg["hidden_dim"],
        use_positional_encoding=False,
        num_blocks=cfg["num_blocks"],
        fusion_type="cross_hyena",
        num_queries=len(QUERY_TRACKS),
        num_context_tracks=len(CONTEXT_TRACKS),
        context_mode="all-others",
    ).to(device)


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
    )


def build_scheduler(optimizer, cfg, epochs: int):
    if cfg["scheduler"] == "none":
        return None
    warm = min(int(cfg["warmup_epochs"]), max(0, epochs - 1))
    if warm > 0:
        linear = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warm), eta_min=cfg["min_lr"]
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[linear, cosine], milestones=[warm]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=cfg["min_lr"]
    )


# ---------------------------------------------------------------------------
# One config
# ---------------------------------------------------------------------------

def run_config(name: str, cfg: dict, args, device, out_root: Path) -> dict:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg_dir = out_root / "runs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    history_path = cfg_dir / f"{stamp}_{name}_history.jsonl"
    history_path.unlink(missing_ok=True)

    set_random_seed(int(cfg["seed"]))
    frame, train_idx, val_idx, test_idx, info = build_split(args, cfg)
    print(
        f"\n=== [{name}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"(split={info['split_case']}) target_len={cfg['target_length']}"
    )
    bed_dir = cfg_dir / f"{stamp}_{name}"
    write_train_val_beds(
        frame, list(train_idx), list(val_idx),
        output_dir=bed_dir, timestamp="", tag=name, test_indices=list(test_idx),
    )

    # deterministic per-region val/test seeds, aligned to sequential loader order
    val_seeds = np.array(
        [region_seed(str(frame.iloc[i]["chr"]), int(frame.iloc[i]["start_expanded"]), 1000 + int(cfg["seed"]))
         for i in val_idx], dtype=np.int64,
    )
    test_seeds = np.array(
        [region_seed(str(frame.iloc[i]["chr"]), int(frame.iloc[i]["start_expanded"]), 2000 + int(cfg["seed"]))
         for i in test_idx], dtype=np.int64,
    )

    train_loader = make_loader(frame, train_idx, args, cfg, shuffle=True, augment_rc=args.augment_rc)
    val_loader = make_loader(frame, val_idx, args, cfg, shuffle=False, augment_rc=False)
    test_loader = make_loader(frame, test_idx, args, cfg, shuffle=False, augment_rc=False) if len(test_idx) else None

    model = build_model(cfg, cfg["target_length"], device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, args.epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if (args.amp and device.type == "cuda") else torch.float32

    best = {"val_total_loss": float("inf")}
    best_state = None
    best_epoch = 0
    epoch_seconds: list[float] = []
    peak_mb = 0.0
    nq, nc = len(QUERY_TRACKS), len(CONTEXT_TRACKS)

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        sums = {n: 0.0 for n in QUERY_TRACKS}
        total_sum, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        for batch in train_loader:
            queries = [t.to(device, non_blocking=True) for t in batch[:nq]]
            context = [t.to(device, non_blocking=True) for t in batch[nq:nq + nc]]
            seq = batch[nq + nc].to(device, non_blocking=True)
            base_ids = batch[nq + nc + 1]

            bsz = base_ids.size(0)
            train_seeds = np.random.randint(0, 2**31 - 1, size=bsz)
            hide, loss_mask = pair_masks(base_ids, train_seeds, cfg["mask_fraction"], cfg["neighborhood_bp"])
            hide, loss_mask = hide.to(device, non_blocking=True), loss_mask.to(device, non_blocking=True)

            # whole-track (modality) dropout, per sample, re-drawn every step
            track_present = [
                (torch.rand(bsz, 1, 1, device=device) >= cfg[f"drop_p_{t}"]).float()
                for t in CONTEXT_TRACKS
            ]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                preds = model(queries, seq, context, position_mask=hide, track_present=track_present)
                per_head = [masked_mse(p, q, loss_mask) for p, q in zip(preds, queries)]
                loss = torch.stack(per_head).mean() / cfg["grad_accum"]

            scaler.scale(loss).backward()
            accum += 1
            if accum % cfg["grad_accum"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            seen += bsz
            total_sum += float(loss.item()) * bsz * cfg["grad_accum"]
            for n, l in zip(QUERY_TRACKS, per_head):
                sums[n] += float(l.item()) * bsz

        if accum % cfg["grad_accum"] != 0:          # flush the tail
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val = evaluate(model, val_loader, val_seeds, cfg, device, args.amp, desc="val")
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        secs = time.time() - t_epoch
        epoch_seconds.append(secs)
        if device.type == "cuda":
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated(device) / 2**20)

        is_best = val["val_total_loss"] < best["val_total_loss"]
        if is_best:
            best = dict(val)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        win = seen + val["windows"]
        print(
            f"[{name}] ep{epoch:02d}/{args.epochs} "
            f"train={total_sum / max(seen, 1):.2f} "
            f"(5mc={sums['5mc'] / max(seen, 1):.2f}, 5hmc={sums['5hmc'] / max(seen, 1):.2f}) | "
            f"val={val['val_total_loss']:.3f} "
            f"R2 5mc={val['val_5mc_r2']:.4f} 5hmc={val['val_5hmc_r2']:.4f} | "
            f"lr={lr_now:.2e} | {secs:.0f}s ({win / secs:.1f} win/s) | {peak_mb:.0f}MB"
        )
        with open(history_path, "a") as fh:
            fh.write(json.dumps({
                "config": name, "epoch": epoch, "lr": lr_now,
                "train_total_loss": total_sum / max(seen, 1),
                **{f"train_{n}_loss": sums[n] / max(seen, 1) for n in QUERY_TRACKS},
                **val, "epoch_seconds": round(secs, 1),
                "windows_per_s": round(win / secs, 2),
                "peak_mb": round(peak_mb, 1),
                "is_best": bool(is_best),
            }) + "\n")

    # ---- test, once, from the best-val weights -------------------------------
    test = {}
    if test_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        test = evaluate(model, test_loader, test_seeds, cfg, device, args.amp, desc="test")

    if args.save_ckpt and best_state is not None:
        torch.save({"model_state_dict": best_state, "config": cfg, "name": name},
                   cfg_dir / f"{stamp}_{name}_best.pt")

    record = {
        "config": name, "status": "ok",
        "lr": cfg["learning_rate"], "batch_size": cfg["batch_size"], "grad_accum": cfg["grad_accum"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum"],
        "hidden_dim": cfg["hidden_dim"], "num_blocks": cfg["num_blocks"],
        "mask_fraction": cfg["mask_fraction"], "neighborhood_bp": cfg["neighborhood_bp"],
        "weight_decay": cfg["weight_decay"], "min_lr": cfg["min_lr"],
        "scheduler": cfg["scheduler"], "warmup_epochs": cfg["warmup_epochs"],
        "target_length": cfg["target_length"],
        "drop_p_atac": cfg["drop_p_atac"], "drop_p_rna": cfg["drop_p_rna"],
        "epochs": args.epochs, "seed": cfg["seed"], "n_params": n_params,
        "train_windows": int(len(train_idx)), "val_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)), "split_case": info["split_case"],
        "best_epoch": best_epoch,
        **best, **test,
        "score": 0.5 * (best.get("val_5mc_r2", float("nan")) + best.get("val_5hmc_r2", float("nan"))),
        "epoch_seconds_mean": float(np.mean(epoch_seconds)) if epoch_seconds else float("nan"),
        "peak_mb": round(peak_mb, 1),
        "timestamp": stamp, "history": str(history_path),
    }
    print(
        f"[{name}] BEST ep{best_epoch} val={best['val_total_loss']:.3f} "
        f"(score={record['score']:.4f}) | "
        f"TEST {test.get('test_total_loss', float('nan')):.3f} "
        f"R2 5mc={test.get('test_5mc_r2', float('nan')):.4f} "
        f"5hmc={test.get('test_5hmc_r2', float('nan')):.4f}"
    )
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--dmr-csv", default="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed")
    g.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    g.add_argument("--m5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz")
    g.add_argument("--hm5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz")
    g.add_argument("--atac-bw", default="/data1st1/junyi/methdata/atac/AMY_MC_track.bw")
    g.add_argument("--rna-coverage-bw", default="/data2st1/junyi/output/sn0615/BULK_AMY/AMY_MC.bw")
    g.add_argument("--test-bed", default="output/holdout_testchroms/cpg_test.bed",
                   help="Held-out-chromosome BED; pass '' to fall back to auto 80/10/10.")
    g.add_argument("--proxy-size", type=int, default=20000,
                   help="Random genome-wide subset size for screening (0 = full data). "
                        "20k keeps ~570 optimizer steps inside a 5-epoch screen (production "
                        "does ~444 steps per epoch on the full 80k set).")
    g.add_argument("--proxy-seed", type=int, default=7)
    g.add_argument("--train-ratio", type=float, default=0.8)

    g = p.add_argument_group("run")
    g.add_argument("--epochs", type=int, default=5)
    g.add_argument("--num-workers", type=int, default=8)
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--augment-rc", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--out-dir", default="output/parametest")
    g.add_argument("--tag", default="")
    g.add_argument("--force-scheduler", choices=["none", "cosine", ""], default="none",
                   help="Override every config's lr schedule. Default 'none' (constant lr): a "
                        "short screening run must not anneal away its own learning rate. "
                        "Pass '' to honour each config, or 'cosine' for the production schedule.")
    g.add_argument("--shard", default="0/1", help="i/n -- run every n-th config starting at i.")
    g.add_argument("--only", default="", help="Comma-separated config names.")
    g.add_argument("--list-configs", action="store_true")
    g.add_argument("--save-ckpt", action="store_true")
    g.add_argument("--smoke", action="store_true", help="480 windows, 1 epoch, baseline only.")
    g.add_argument("--seed", type=int, default=None, help="Override every config's seed.")
    add_clip_at_zero_argument(p)
    return p.parse_args()


def select_configs(args) -> list[tuple[str, dict]]:
    grid = GRID
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - {n for n, _ in grid}
        if unknown:
            raise SystemExit(f"unknown config(s): {sorted(unknown)}; see --list-configs")
        grid = [(n, o) for n, o in grid if n in wanted]
    i, n = (int(x) for x in args.shard.split("/"))
    if not (0 <= i < n):
        raise SystemExit(f"bad --shard {args.shard}")
    grid = [c for k, c in enumerate(grid) if k % n == i]

    out = []
    for name, overrides in grid:
        cfg = dict(DEFAULTS)
        cfg.update(overrides)
        if args.force_scheduler:
            cfg["scheduler"] = args.force_scheduler
        if args.seed is not None:
            cfg["seed"] = args.seed
        out.append((name, cfg))
    return out


def main() -> int:
    args = parse_args()
    if args.list_configs:
        for name, overrides in GRID:
            print(f"{name:14s} {json.dumps(overrides)}")
        return 0

    if args.smoke:
        args.proxy_size, args.epochs, args.num_workers = 480, 1, 0
        args.only = args.only or "baseline"
        args.out_dir = "/tmp/parametest_smoke"

    if torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{get_freest_gpu()}")
    else:
        device = torch.device("cpu")

    out_root = Path(args.out_dir) / (args.tag or datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    out_root.mkdir(parents=True, exist_ok=True)

    configs = select_configs(args)
    print("=" * 78)
    print("parametest: standalone screening run")
    print(f"  device        : {device}  ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")
    print(f"  epochs        : {args.epochs}")
    print(f"  proxy size    : {args.proxy_size or 'ALL'} (seed {args.proxy_seed})")
    print(f"  split         : test bed = {args.test_bed or 'auto 80/10/10'}")
    print(f"  tracks        : query={QUERY_TRACKS} context={CONTEXT_TRACKS} (all-others)")
    print(f"  RC augment    : {args.augment_rc} (train only)")
    print(f"  configs       : {len(configs)} -> {[n for n, _ in configs]}")
    print(f"  output        : {out_root}")
    print("=" * 78)
    (out_root / "grid.json").write_text(json.dumps(
        {"defaults": DEFAULTS, "grid": [{"name": n, "config": c} for n, c in configs],
         "args": vars(args)}, indent=2))

    runs_path = out_root / "runs.jsonl"
    records = []
    for name, cfg in configs:
        try:
            rec = run_config(name, cfg, args, device, out_root)
        except Exception as exc:                     # keep the shard alive
            import traceback
            traceback.print_exc()
            rec = {"config": name, "status": f"error: {type(exc).__name__}: {exc}"}
        records.append(rec)
        with open(runs_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_root / "summary.csv", index=False)
        if "score" in df.columns:
            cols = [c for c in ["config", "score", "best_epoch", "val_total_loss",
                                "val_5mc_r2", "val_5hmc_r2", "test_total_loss",
                                "test_5mc_r2", "test_5hmc_r2", "epoch_seconds_mean",
                                "status"] if c in df.columns]
            print("\n" + df.sort_values("score", ascending=False)[cols].to_string(index=False))
    print(f"\n[done] {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
