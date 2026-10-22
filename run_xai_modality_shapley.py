#!/usr/bin/env python
"""Exact modality-level Shapley values + Shapley interaction indices.

Reads a checkpoint written by ``run_multiquery_pretraining.py`` (``.pt``) and
its result file (``.json``), rebuilds *exactly* the same model and the same
train/val/test split, then explains one head's masked-reconstruction prediction
with a full 2^n coalition enumeration:

  * ``phi_a``   Shapley value of every modality player
  * ``I_ab``    Grabisch-Roubens pairwise Shapley interaction index
  * ``F_ab``    pure 2x2 factorial interaction ``v(ab) - v(a) - v(b) + v(empty)``
  * ``m(S)``    Harsanyi dividends -- the ONLY strictly additive decomposition

All quantities are in the *target's own units* (for the methylation runs the
bedGraph tracks are percentages, so phi is in 5hmC percentage points, 0-100).

4 players -> 16 forward passes, 5 players -> 32, per *window* (not per CpG):
the value function returns a dense ``(N, seq_len)`` prediction tensor, so one
forward pass scores every CpG of a window at once.

Two properties of the model shape this script; both were verified against the
2026-10-13 AMY_MC cCRE checkpoint, not assumed:

1. Co-located signal is structurally invisible at a scored position.
   ``models.py`` applies the SAME position mask to every signal track
   (queries included) *before* the blocks, so at a masked position the hidden
   state is the learned ``[POS-MASK]`` token regardless of the track value and
   ``d pred(i) / d track(i) == 0`` exactly.  Every attribution produced here is
   therefore a *neighbourhood* attribution ("the modality elsewhere in the
   16 kb window"), never a co-located one.  ``--positions cpg_forward`` keeps
   only forward-strand CpG C positions (the G half of the mask predicts a
   constant 0 target and would pollute every average).

2. ``[MODALITY-MASK]`` tokens are NOT a usable absence operator on this
   checkpoint.  Runs trained with ``--drop-p-* 0.0`` never gradient-update
   ``query_mask_tokens`` / ``context_track_mask_tokens`` / ``sequence_mask_token``
   (measured absmax == 0), so ``*_present = 0`` substitutes a zero vector, not a
   learned "missing modality" state.  Absence is expressed in input space
   instead (donor window / within-window permutation / cross-region swap /
   zero / mean), which is well defined for every checkpoint.  ``--absence token``
   is available but hard-fails unless the run really trained those tokens.

Math notes (kept separate on purpose -- do NOT add phi and I together):

  phi_a = sum_{S subset P\\{a}} |S|!(n-|S|-1)!/n! * [v(S+a) - v(S)]
        efficiency: sum_a phi_a == v(P) - v(empty)          (asserted)

  I_ab  = sum_{S subset P\\{a,b}} |S|!(n-|S|-2)!/(n-1)! *
          [v(S+ab) - v(S+a) - v(S+b) + v(S)]
        signed redundancy/synergy; NOT additive with phi.

  m(S)  = sum_{T subset S} (-1)^{|S|-|T|} v(T)
        v(P) - v(empty) == sum_{S != empty} m(S)           (asserted)
        => report main effects m({a}), pure pair terms m({a,b}) and m(|S|>=3).

Usage:
  # 0) sanity: prove the checkpoint + split were rebuilt correctly
  python run_xai_modality_shapley.py verify \
      --checkpoint <..._best_80137.pt> --results-json <..._results.json>

  # 1) exact Shapley + interactions on 1500 held-out test windows
  python run_xai_modality_shapley.py shapley \
      --checkpoint <..._best_80137.pt> --results-json <..._results.json> \
      --head 5hmc --split test --n-windows 1500 \
      --players seq,other,atac,rna,self --absence donor \
      --device cuda:5 --batch-size 8 \
      --out-dir output/AMY_MC/cCRE_cpg/xai
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import (
    LazyM5cSequenceAtacRnaDataset,
    find_cpg_candidate_positions,
    find_forward_cpg_positions,
    generate_pretraining_cpg_mask,
    load_data,
    resolve_train_val_test_split,
)
from models import MultiQueryMaskedPretrainingModel
from run_multiquery_pretraining import (
    MultiQueryDataset,
    evaluate,
    load_model_from_checkpoint,
)
from utils import ensure_parent_dir, get_freest_gpu, set_random_seed

VERSION = "1.0"

# Flat per-track order used by MultiQueryDataset: query tracks first, then context.
TRACK_INDEX = {"5mc": 0, "5hmc": 1, "atac": 2, "rna": 3}
CANONICAL_TRACK_ORDER = ["5mc", "5hmc", "atac", "rna"]

# args that must agree between the checkpoint and the results JSON
ARCH_KEYS = (
    "target_length", "hidden_dim", "num_blocks", "fusion_type", "context_mode",
    "use_positional_encoding", "query_tracks",
)
SPLIT_KEYS = ("dmr_csv", "genome_fasta", "train_ratio", "test_bed", "seed",
              "mask_fraction", "mask_mode")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def first_path(value):
    """Return the first path of a str / list / None CLI value."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


def sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def resolve_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return torch.device("cuda:0")
    try:
        return torch.device(f"cuda:{get_freest_gpu()}")
    except Exception:
        return torch.device("cuda:0")


def fmt_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def append(store: dict, key: str, value: np.ndarray) -> None:
    store.setdefault(key, []).append(value)


# ---------------------------------------------------------------------------
# Inputs: checkpoint + results JSON
# ---------------------------------------------------------------------------

def load_inputs(checkpoint_path: str, json_path: str):
    """Load .pt and .json, return (checkpoint, payload, entry, merged_args)."""
    if not Path(checkpoint_path).is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if not Path(json_path).is_file():
        raise SystemExit(f"results json not found: {json_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"{checkpoint_path} has no 'model_state_dict'")
    with open(json_path) as fh:
        payload = json.load(fh)

    pt_args = dict(checkpoint.get("args") or {})
    json_args = dict(payload.get("args") or {})
    results = payload.get("results") or []
    if not results:
        raise SystemExit(f"{json_path} has no 'results' entries")

    num_dmrs = int(checkpoint.get("num_dmrs") or 0)
    entry = None
    for cand in results:
        if num_dmrs and int(cand.get("num_dmrs") or -1) == num_dmrs:
            entry = cand
            break
    if entry is None:
        entry = max(results, key=lambda r: int(r.get("num_dmrs") or -1))

    problems = []
    for key in ARCH_KEYS + SPLIT_KEYS:
        a, b = pt_args.get(key), json_args.get(key)
        if key in pt_args and key in json_args and a != b:
            problems.append(f"  {key}: pt={a!r}  json={b!r}")
    if problems:
        raise SystemExit(
            "checkpoint and results JSON disagree on architecture/split:\n"
            + "\n".join(problems)
        )

    merged = {**json_args, **pt_args}
    merged["_checkpoint_path"] = str(checkpoint_path)
    merged["_json_path"] = str(json_path)
    merged["_entry"] = entry
    return checkpoint, payload, entry, merged


def build_model(merged: dict, entry: dict, device: torch.device):
    """Rebuild the exact architecture from the checkpoint args and load weights."""
    if int(merged.get("target_length", -1)) != 16384:
        raise SystemExit(
            "target_length must be 16384: the Hyena filters are instantiated with "
            "lengths derived from seq_len, so the model is length-locked."
        )
    query_tracks = list(entry["query_tracks"])
    context_tracks = list(entry["context_tracks"])

    model = MultiQueryMaskedPretrainingModel(
        seq_len=int(merged["target_length"]),
        hidden_dim=int(merged["hidden_dim"]),
        use_positional_encoding=bool(merged.get("use_positional_encoding", False)),
        num_blocks=int(merged["num_blocks"]),
        fusion_type=str(merged["fusion_type"]),
        num_queries=len(query_tracks),
        num_context_tracks=len(context_tracks),
        context_mode=str(merged["context_mode"]),
    ).to(device)

    epoch, _ = load_model_from_checkpoint(model, merged["_checkpoint_path"], device)
    model.eval()
    model.float()
    for param in model.parameters():
        param.requires_grad_(False)
    print(
        f"[model] loaded epoch {epoch} | "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params + "
        f"{sum(b.numel() for b in model.buffers()) / 1e6:.2f}M buffers | "
        f"queries={query_tracks} context={context_tracks} | "
        f"context_mode={model.context_mode} | device={device}"
    )
    if len(query_tracks) + len(context_tracks) != 4:
        raise SystemExit(
            "this script assumes the full 4-track layout "
            "(5mc, 5hmc, atac, rna); got "
            f"{query_tracks + context_tracks}"
        )
    return model, query_tracks, context_tracks


def rebuild_split(merged: dict, entry: dict):
    """Recreate the exact same train/val/test split as training (read-only)."""
    args = argparse.Namespace(**merged)
    args.use_m5c = True
    args.sample_sizes = [int(entry["num_dmrs"])]

    df_dmr, *_ = load_data(args, lazy=True)
    num_dmrs = int(entry["num_dmrs"])
    if len(df_dmr) > num_dmrs:
        df_dmr = df_dmr.iloc[:num_dmrs].copy().reset_index(drop=True)
    if len(df_dmr) != num_dmrs:
        raise SystemExit(f"DMR table has {len(df_dmr)} rows but the run used {num_dmrs}")

    split_df = df_dmr.copy().reset_index().rename(columns={"index": "original_idx"})
    split_df["chr"] = split_df["chr"].astype(str)
    split_df["start_expanded"] = split_df["start_expanded"].astype(int)
    split_df["end_expanded"] = split_df["end_expanded"].astype(int)

    train_idx, val_idx, test_idx, info = resolve_train_val_test_split(
        split_df,
        train_ratio=float(merged["train_ratio"]),
        test_bed_path=merged.get("test_bed"),
    )

    expected = entry.get("split_info") or {}
    checks = {
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "test_windows": len(test_idx),
    }
    bad = {k: (v, expected.get(k)) for k, v in checks.items()
           if expected.get(k) is not None and int(expected[k]) != v}
    if bad:
        raise SystemExit(
            "rebuilt split does not match the results JSON: "
            + ", ".join(f"{k} rebuilt={v[0]} json={v[1]}" for k, v in bad.items())
            + "\n  (check dmr_csv / train_ratio / test_bed in the checkpoint args)"
        )
    print(f"[split] case={info['split_case']} train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)}")
    return split_df, {"train": train_idx, "val": val_idx, "test": test_idx}


def verify_test_bed(merged: dict, split_df: pd.DataFrame, test_idx: np.ndarray,
                    bed_path: str | None):
    """Compare the rebuilt test windows with the *_test_regions.bed on disk."""
    if not bed_path:
        default = Path(merged["_json_path"]).parent / (
            f"{merged.get('timestamp', '')}_test_regions.bed"
        )
        bed_path = str(default) if default.is_file() else None
    if not bed_path:
        print("[bed] no test_regions.bed found; skipping coordinate check")
        return
    bed = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    rebuilt = split_df.iloc[test_idx].reset_index(drop=True)
    if len(bed) != len(rebuilt):
        raise SystemExit(
            f"test BED has {len(bed)} rows but the rebuilt split has {len(rebuilt)}"
        )
    same = (
        (bed.iloc[:, 0].astype(str).to_numpy() == rebuilt["chr"].astype(str).to_numpy())
        & (bed.iloc[:, 1].to_numpy(dtype=np.int64)
           == rebuilt["start_expanded"].to_numpy(dtype=np.int64))
        & (bed.iloc[:, 2].to_numpy(dtype=np.int64)
           == rebuilt["end_expanded"].to_numpy(dtype=np.int64))
    )
    if not same.all():
        first = int(np.flatnonzero(~same)[0])
        raise SystemExit(
            f"test BED mismatch at row {first}: "
            f"bed={bed.iloc[first, :3].tolist()} "
            f"rebuilt={rebuilt.iloc[first][['chr', 'start_expanded', 'end_expanded']].to_dict()}"
        )
    print(f"[bed] {len(bed)} test windows match {bed_path}")


# ---------------------------------------------------------------------------
# Datasets / loaders
# ---------------------------------------------------------------------------

def make_loader(merged: dict, split_df: pd.DataFrame, indices, track_order: list[str],
                batch_size: int, num_workers: int, overrides: dict | None = None):
    """Build the lazy dataset + MultiQueryDataset + DataLoader (no side effects)."""
    overrides = overrides or {}
    paths = {
        "5mc": overrides.get("5mc", first_path(merged.get("m5c_bedgraph"))),
        "5hmc": overrides.get("5hmc", first_path(merged.get("hm5c_bedgraph"))),
        "atac": overrides.get("atac", first_path(merged.get("atac_bw"))),
        "rna": overrides.get("rna", first_path(merged.get("rna_coverage_bw"))),
    }
    base = LazyM5cSequenceAtacRnaDataset(
        indices=list(indices),
        df_dmr=split_df,
        genome_fasta=str(merged["genome_fasta"]),
        m5c_bedgraph=str(paths["5mc"]),
        hm5c_bedgraph=str(paths["5hmc"]),
        atac_bw_path=str(paths["atac"]) if paths["atac"] else None,
        target_length=int(merged["target_length"]),
        mask_mode=str(merged["mask_mode"]),
        atac_scaling=str(merged.get("atac_scaling", "none")),
        rna_bw_path=str(paths["rna"]) if paths["rna"] else None,
        rna_scaling=str(merged.get("rna_scaling", "minmax")),
        augment_rc=False,
        clip_at_zero=bool(merged.get("clip_at_zero", False)),
    )
    dataset = MultiQueryDataset(base, list(track_order))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def donor_source_indices(ctx, split_name: str, selected: np.ndarray) -> np.ndarray:
    """Windows used as 'this modality elsewhere' donors.

    Prefer the OTHER splits: the train/val/test groups are non-overlapping by
    construction, so a donor can never be the same locus as the explained window
    (which would make the intervention nearly a no-op).  When the run only has a
    single split, fall back to that split and let the caller exclude self.
    """
    other = [name for name in ("test", "val", "train") if name != split_name]
    parts = [np.asarray(ctx["split_idx"][name], dtype=np.int64) for name in other]
    source = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if source.size == 0:
        source = np.asarray(ctx["split_idx"][split_name], dtype=np.int64)
    return source


def sample_window_indices(indices: np.ndarray, n_windows: int, seed: int) -> np.ndarray:
    """Deterministic subset of rows, returned in ascending order (genomic order)."""
    indices = np.asarray(indices, dtype=np.int64)
    if n_windows <= 0 or n_windows >= len(indices):
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(indices), size=int(n_windows), replace=False)
    return np.sort(indices[np.sort(picked)])


def split_batch(batch, n_tracks: int):
    """Unpack a MultiQueryDataset batch into (tracks, sequence_onehot, base_ids)."""
    if len(batch) != n_tracks + 2:
        raise SystemExit(f"unexpected batch size {len(batch)} (expected {n_tracks + 2})")
    tracks = [batch[i] for i in range(n_tracks)]
    return tracks, batch[n_tracks], batch[n_tracks + 1]


# ---------------------------------------------------------------------------
# Absence operators (input-space interventions)
# ---------------------------------------------------------------------------

def sequence_absence(sequence: torch.Tensor, base_ids: torch.Tensor,
                     rng: np.random.Generator, mode: str = "at") -> torch.Tensor:
    """CpG-preserving sequence shuffle.

    ``at``  : keep every C and G, permute only A/T among A/T slots.  The CpG
              dinucleotide set -- and therefore CpG density -- is unchanged.
    ``cpg`` : keep only C/G that form a CpG, permute the rest (weaker control).
    """
    out = sequence.clone()
    for b in range(sequence.shape[0]):
        ids = base_ids[b].detach().cpu().numpy().astype(np.int64)
        if mode == "at":
            movable = np.isin(ids, (0, 3))
        elif mode == "cpg":
            cpg = find_cpg_candidate_positions(base_ids[b:b + 1])[0].cpu().numpy()
            movable = ~cpg
        else:
            raise SystemExit(f"unknown --seq-shuffle mode: {mode}")
        pos = np.flatnonzero(movable)
        if pos.size < 2:
            continue
        shuffled = ids.copy()
        shuffled[pos] = ids[pos[rng.permutation(pos.size)]]
        out[b] = torch.nn.functional.one_hot(
            torch.from_numpy(shuffled).to(sequence.device), num_classes=4
        ).float()
    return out


def permuted_track(track: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Permute a track's values within each window (support-preserving).

    Only non-zero entries move, so a CG bedGraph keeps its exact C support and
    its exact marginal value distribution; the locus-specific correspondence to
    the sequence / target is destroyed.
    """
    out = track.clone()
    for b in range(track.shape[0]):
        nonzero = torch.nonzero(track[b, :, 0] != 0, as_tuple=False).squeeze(-1)
        n = int(nonzero.numel())
        if n < 2:
            continue
        perm = nonzero[torch.from_numpy(rng.permutation(n)).to(track.device)]
        out[b, nonzero, :] = track[b, perm, :]
    return out


class DonorPool:
    """A small cache of real windows used as 'this modality elsewhere' donors."""

    def __init__(self, tensors: dict[str, torch.Tensor], means: dict[str, float],
                 indices: np.ndarray | None = None):
        self.tensors = tensors
        self.means = means
        self.indices = indices
        self.size = len(next(iter(tensors.values()))) if tensors else 0

    @classmethod
    def build(cls, merged, split_df, pool_indices, track_order, n_tracks):
        if len(pool_indices) == 0:
            return cls({}, {})
        pool_indices = np.asarray(pool_indices, dtype=np.int64)
        _, loader = make_loader(
            merged, split_df, pool_indices, track_order,
            batch_size=min(16, len(pool_indices)), num_workers=0,
        )
        chunks: dict[str, list[torch.Tensor]] = {name: [] for name in track_order}
        for batch in loader:
            tracks, _seq, _base = split_batch(batch, n_tracks)
            for name, tensor in zip(track_order, tracks):
                chunks[name].append(tensor.cpu())
        tensors = {name: torch.cat(chunks[name], dim=0) for name in track_order}
        means = {name: float(t.numpy().mean()) for name, t in tensors.items()}
        print(f"[donor] pool of {len(next(iter(tensors.values())))} windows | "
              + ", ".join(f"{k}={v:.4f}" for k, v in means.items()))
        return cls(tensors, means, pool_indices)

    def gather(self, name: str, donor_ids: np.ndarray, device) -> torch.Tensor:
        return self.tensors[name][donor_ids].to(device)


class Player:
    """One participant of the coalition game."""

    __slots__ = ("name", "slot", "kind", "label")

    def __init__(self, name: str, slot: int | None, kind: str, label: str):
        self.name = name
        self.slot = slot
        self.kind = kind          # "seq" | "track"
        self.label = label

    def __repr__(self):
        return f"Player({self.label}, slot={self.slot}, kind={self.kind})"


def resolve_players(tokens: list[str], head: str, query_tracks: list[str],
                    context_tracks: list[str], context_specs) -> list[Player]:
    """Map user tokens (seq / other / self / <track>) to concrete player slots."""
    head_idx = query_tracks.index(head)
    others = [t for t in query_tracks if t != head]
    players: list[Player] = []
    seen: set[int | None] = set()

    for raw in tokens:
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok == "seq":
            name, slot, kind, label = "seq", None, "seq", "seq"
        elif tok in ("self", "own", "query"):
            name, slot, kind, label = head, head_idx, "track", f"{head}(self)"
        elif tok in ("other", "ctx"):
            if not others:
                raise SystemExit("'other' requested but the run has a single query track")
            name = others[0]
            slot, kind, label = TRACK_INDEX[name], "track", name
        elif tok in ("5mc", "5hmc", "atac", "rna"):
            name, slot, kind, label = tok, TRACK_INDEX[tok], "track", tok
        else:
            raise SystemExit(
                f"unknown player {raw!r}; use seq / other / self / 5mc / 5hmc / atac / rna"
            )

        if slot in seen:
            raise SystemExit(
                f"player {label!r} resolves to an already-used slot — drop the duplicate "
                f"(for --head {head} the 'other' query is "
                f"{others[0] if others else 'n/a'})."
            )
        seen.add(slot)

        # A track player must really feed this head: either as one of its context
        # tracks, or as its own query trunk (the 'self' neighbourhood channel).
        if kind == "track" and slot != head_idx:
            spec = list(context_specs[head_idx])
            if slot not in spec:
                raise SystemExit(
                    f"player {label!r} (slot {slot}) is not in the context of head "
                    f"{head!r} (context slots {spec}); the query streams must not "
                    f"read each other in context_mode='shared'."
                )
        players.append(Player(name, slot, kind, label))

    if not players:
        raise SystemExit("--players is empty")
    return players


# ---------------------------------------------------------------------------
# Coalition enumeration
# ---------------------------------------------------------------------------

def prepare_absences(players: list[Player], tracks: list[torch.Tensor],
                     sequence: torch.Tensor, base_ids: torch.Tensor, absence: str,
                     donor_tracks: dict[str, torch.Tensor] | None, pool: DonorPool | None,
                     rng_seq: np.random.Generator, rng_perm: np.random.Generator,
                     seq_mode: str, device) -> dict[int, torch.Tensor]:
    """Precompute, once per batch, the replacement tensor of every player."""
    absent: dict[int, torch.Tensor] = {}
    for i, player in enumerate(players):
        if player.kind == "seq":
            absent[i] = sequence_absence(sequence, base_ids, rng_seq, mode=seq_mode)
            continue
        original = tracks[player.slot]
        if absence in ("donor", "cross-region"):
            if donor_tracks is None or player.name not in donor_tracks:
                raise SystemExit(f"absence={absence} needs a donor tensor for {player.name}")
            absent[i] = donor_tracks[player.name].to(device)
        elif absence == "permute":
            absent[i] = permuted_track(original, rng_perm)
        elif absence == "mean":
            if pool is None or player.name not in pool.means:
                raise SystemExit("absence=mean needs a non-empty donor pool")
            absent[i] = torch.full_like(original, float(pool.means[player.name]))
        elif absence == "zero":
            absent[i] = torch.zeros_like(original)
        elif absence == "token":
            absent[i] = original  # substituted through presence flags instead
        else:
            raise SystemExit(f"unknown --absence: {absence}")
    return absent


@torch.no_grad()
def run_coalitions(model, players: list[Player], absent: dict[int, torch.Tensor],
                   tracks: list[torch.Tensor], sequence: torch.Tensor,
                   position_mask: torch.Tensor, head_idx: int, n_queries: int,
                   n_context: int, absence: str,
                   presence: dict[str, torch.Tensor] | None) -> dict[int, torch.Tensor]:
    """Evaluate v(S) for every subset S; returns dict[bitmask] -> (B, L)."""
    values: dict[int, torch.Tensor] = {}
    for coalition in range(1 << len(players)):
        if absence == "token":
            q_present = [presence["query"][i].clone() for i in range(n_queries)]
            t_present = [presence["track"][k].clone() for k in range(n_context)]
            s_present = presence["seq"].clone()
            for i, player in enumerate(players):
                if (coalition >> i) & 1:
                    continue
                if player.kind == "seq":
                    s_present = torch.zeros_like(s_present)
                elif player.slot < n_queries:
                    q_present[player.slot] = torch.zeros_like(q_present[player.slot])
                else:
                    k = player.slot - n_queries
                    t_present[k] = torch.zeros_like(t_present[k])
            preds = model(
                list(tracks[:n_queries]), sequence, list(tracks[n_queries:]),
                position_mask=position_mask, query_present=q_present,
                track_present=t_present, sequence_present=s_present,
            )
        else:
            mutated_tracks = list(tracks)
            mutated_seq = sequence
            for i, player in enumerate(players):
                if (coalition >> i) & 1:
                    continue
                if player.kind == "seq":
                    mutated_seq = absent[i]
                else:
                    mutated_tracks[player.slot] = absent[i]
            preds = model(
                mutated_tracks[:n_queries], mutated_seq, mutated_tracks[n_queries:],
                position_mask=position_mask,
            )
        values[coalition] = preds[head_idx].squeeze(-1)
    return values


# ---------------------------------------------------------------------------
# Game-theoretic quantities (vectorised over positions)
# ---------------------------------------------------------------------------

def shapley_values(values: dict[int, torch.Tensor], n_players: int) -> list[torch.Tensor]:
    weights = [factorial(s) * factorial(n_players - s - 1) / factorial(n_players)
               for s in range(n_players)]
    phi: list[torch.Tensor] = []
    for a in range(n_players):
        bit = 1 << a
        acc = torch.zeros_like(values[0])
        for coalition in range(1 << n_players):
            if coalition & bit:
                continue
            acc = acc + weights[coalition.bit_count()] * (
                values[coalition | bit] - values[coalition]
            )
        phi.append(acc)
    return phi


def pairwise_shapley_interaction(values: dict[int, torch.Tensor],
                                 n_players: int) -> dict[tuple[int, int], torch.Tensor]:
    out: dict[tuple[int, int], torch.Tensor] = {}
    for a, b in combinations(range(n_players), 2):
        ba, bb = 1 << a, 1 << b
        acc = torch.zeros_like(values[0])
        for coalition in range(1 << n_players):
            if coalition & (ba | bb):
                continue
            s = coalition.bit_count()
            weight = factorial(s) * factorial(n_players - s - 2) / factorial(n_players - 1)
            acc = acc + weight * (
                values[coalition | ba | bb]
                - values[coalition | ba]
                - values[coalition | bb]
                + values[coalition]
            )
        out[(a, b)] = acc
    return out


def factorial_interaction(values: dict[int, torch.Tensor],
                          n_players: int) -> dict[tuple[int, int], torch.Tensor]:
    """Pure 2x2 interaction with every other player absent."""
    out = {}
    for a, b in combinations(range(n_players), 2):
        ba, bb = 1 << a, 1 << b
        out[(a, b)] = values[ba | bb] - values[ba] - values[bb] + values[0]
    return out


def harsanyi_dividends(values: dict[int, torch.Tensor],
                       n_players: int) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for subset in range(1 << n_players):
        acc = torch.zeros_like(values[0])
        inner = subset
        while True:
            sign = -1.0 if ((subset.bit_count() - inner.bit_count()) % 2) else 1.0
            acc = acc + sign * values[inner]
            if inner == 0:
                break
            inner = (inner - 1) & subset
        out[subset] = acc
    return out


def scored_positions(base_ids: torch.Tensor, position_mask: torch.Tensor,
                     positions: str) -> torch.Tensor:
    """Boolean (B, L) = masked positions that pass the strand filter."""
    if positions == "cpg_forward":
        cpg = find_forward_cpg_positions(base_ids)
    elif positions == "cpg":
        cpg = find_cpg_candidate_positions(base_ids)
    elif positions == "c":
        cpg = base_ids == 1
    else:
        raise SystemExit(f"unknown --positions: {positions}")
    return position_mask.squeeze(-1).bool() & cpg


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cluster_bootstrap_mean(per_window_sum: np.ndarray, per_window_count: np.ndarray,
                           n_boot: int, seed: int) -> tuple[float, float, float]:
    """Window-level bootstrap CI for a ratio of sums (CpGs are autocorrelated).

    With fewer than two windows the CI is undefined (there is nothing to
    resample), so the bounds come back as NaN while the point estimate is kept.
    """
    total_count = per_window_count.sum()
    if per_window_sum.size == 0 or total_count == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(per_window_sum.sum() / total_count)
    if per_window_sum.size < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = per_window_sum.size
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        denom = per_window_count[idx].sum()
        means[i] = per_window_sum[idx].sum() / denom if denom else np.nan
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra = np.array(pd.Series(a).rank(), dtype=np.float64)
    rb = np.array(pd.Series(b).rank(), dtype=np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------

def prepare(opts):
    checkpoint, _payload, entry, merged = load_inputs(opts.checkpoint, opts.results_json)
    if opts.seed is not None:
        merged["seed"] = int(opts.seed)
    if opts.mask_fraction is not None:
        merged["mask_fraction"] = float(opts.mask_fraction)
    device = resolve_device(opts.device)
    set_random_seed(int(merged["seed"]))
    model, query_tracks, context_tracks = build_model(merged, entry, device)
    track_order = query_tracks + context_tracks
    if track_order != [t for t in CANONICAL_TRACK_ORDER if t in track_order]:
        raise SystemExit(f"unexpected track order {track_order}; slot mapping would break")
    split_df, split_idx = rebuild_split(merged, entry)
    verify_test_bed(merged, split_df, split_idx["test"],
                    getattr(opts, "test_regions_bed", None))
    return {
        "merged": merged, "entry": entry, "model": model, "device": device,
        "query_tracks": query_tracks, "context_tracks": context_tracks,
        "track_order": track_order, "split_df": split_df, "split_idx": split_idx,
    }


# ---------------------------------------------------------------------------
# Command: verify
# ---------------------------------------------------------------------------

def cmd_verify(opts) -> int:
    ctx = prepare(opts)
    model, merged, entry = ctx["model"], ctx["merged"], ctx["entry"]
    split = opts.split
    indices = ctx["split_idx"][split]

    print(f"[verify] running the training-time evaluation on {len(indices)} "
          f"{split} windows ...")
    _, loader = make_loader(
        merged, ctx["split_df"], indices, ctx["track_order"],
        batch_size=opts.batch_size, num_workers=opts.num_workers,
    )
    mask_seed = int(merged["seed"]) + (1000 if split == "val" else 2000)
    metrics = evaluate(
        model, loader, ctx["device"], float(merged["mask_fraction"]),
        ctx["query_tracks"], ctx["context_tracks"],
        amp=(ctx["device"].type == "cuda"), mask_seed=mask_seed,
    )

    rows, ok = [], True
    expected_r2 = (entry.get("test_r2") if split == "test"
                   else entry.get("final_val_r2")) or {}
    expected_rho = (entry.get("test_pearsonr") if split == "test"
                    else entry.get("final_val_pearsonr")) or {}
    for name in ctx["query_tracks"]:
        got = float(metrics[f"val_{name}_r2"])
        exp = expected_r2.get(name)
        delta = abs(got - exp) if exp is not None else float("nan")
        passed = exp is None or delta <= opts.tolerance
        ok &= bool(passed)
        rows.append([
            name, f"{got:.6f}",
            "n/a" if exp is None else f"{exp:.6f}",
            "n/a" if exp is None else f"{delta:.2e}",
            f"{float(metrics[f'val_{name}_pearsonr']):.4f}",
            "n/a" if exp is None else f"{float(expected_rho.get(name, float('nan'))):.4f}",
            "PASS" if passed else "FAIL",
        ])
    print()
    print(fmt_table(rows, ["head", "rebuilt R2", "json R2", "|delta|",
                           "rebuilt r", "json r", "status"]))
    print()
    if not ok:
        print(f"[verify] FAILED (tolerance {opts.tolerance:g}) — the rebuild does not "
              f"match the reported numbers; do not trust downstream attributions.",
              file=sys.stderr)
        return 1
    print(f"[verify] PASSED: checkpoint and split rebuilt exactly "
          f"(tolerance {opts.tolerance:g}).")
    return 0


# ---------------------------------------------------------------------------
# Region selection (single window / locus)
# ---------------------------------------------------------------------------

def parse_locus(spec: str) -> tuple[str, int, int]:
    """Parse ``chr9:119592828-119609211`` (1-based, inclusive, commas ok)."""
    text = str(spec).replace(",", "").replace(" ", "")
    if ":" not in text:
        raise SystemExit(f"--locus {spec!r} must look like chr9:119592828-119609211")
    chrom, coords = text.split(":", 1)
    if "-" not in coords:
        raise SystemExit(f"--locus {spec!r} has no '-start-end' part")
    start_text, end_text = coords.split("-", 1)
    try:
        start, end = int(start_text), int(end_text)
    except ValueError:
        raise SystemExit(f"--locus {spec!r}: start/end must be integers")
    if start > end:
        raise SystemExit(f"--locus {spec!r}: start > end")
    return chrom, start, end


def select_region_windows(split_df: pd.DataFrame, split_indices: np.ndarray,
                          split_name: str, locus: str | None = None,
                          original_idx: int | None = None) -> np.ndarray:
    """Resolve one written region to exactly one window index.

    Matching tiers for ``--locus``: exact expanded window -> window containing the
    locus -> window overlapping the locus.  A tier that yields several windows is
    reported (never silently resolved), because overlapping 16 kb windows are
    common and picking one at random would change the answer.
    """
    indices = np.asarray(split_indices, dtype=np.int64)
    rows = split_df.iloc[indices].reset_index(drop=True)

    if original_idx is not None:
        matches = rows["original_idx"].to_numpy(dtype=np.int64) == int(original_idx)
        hits = indices[matches]
        if hits.size == 1:
            row = rows.loc[matches].iloc[0]
            print(f"[region] original_idx={int(original_idx)} -> "
                  f"{row['chr']}:{int(row['start_expanded'])}-{int(row['end_expanded'])}")
            return hits
        if hits.size > 1:
            raise SystemExit(
                f"--window-original-idx {original_idx} matches {hits.size} rows in "
                f"split={split_name}; the DMR table has duplicate original_idx values"
            )
        exists = int(original_idx) in set(
            split_df["original_idx"].to_numpy(dtype=np.int64)
        )
        raise SystemExit(
            f"--window-original-idx {original_idx} is not in split={split_name}"
            + ("" if not exists else " (it exists in the dataset, try another --split)")
            + "\n  hint: column 4 of *_test_regions.bed is original_idx"
        )

    chrom, start, end = parse_locus(locus)
    chr_mask = rows["chr"].astype(str).to_numpy() == chrom
    if not chr_mask.any():
        available = sorted(set(rows["chr"].astype(str)))
        raise SystemExit(
            f"--locus chromosome {chrom!r} is not in split={split_name}; "
            f"available: {available}"
        )
    starts = rows["start_expanded"].to_numpy(dtype=np.int64)
    ends = rows["end_expanded"].to_numpy(dtype=np.int64)
    tiers = (
        ("exact", chr_mask & (starts == start) & (ends == end)),
        ("containing", chr_mask & (starts <= start) & (ends >= end)),
        ("overlapping", chr_mask & (starts <= end) & (ends >= start)),
    )
    for tier, hit_mask in tiers:
        hits = indices[hit_mask]
        if hits.size == 1:
            row = rows.loc[hit_mask].iloc[0]
            print(f"[region] {tier} match -> {row['chr']}:"
                  f"{int(row['start_expanded'])}-{int(row['end_expanded'])} "
                  f"(original_idx={int(row['original_idx'])})")
            return hits
        if hits.size > 1:
            preview = rows.loc[hit_mask, ["chr", "start_expanded", "end_expanded",
                                          "original_idx"]].head(8)
            raise SystemExit(
                f"--locus {locus!r} {tier}-matches {hits.size} windows in split="
                f"{split_name}; disambiguate with --window-original-idx.\n"
                + preview.to_string(index=False)
            )
    raise SystemExit(
        f"--locus {locus!r} no longer falls inside any window of split={split_name}"
    )


def read_bed_raw(bed_path: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a BED into {chrom: [(start, end), ...]} WITHOUT merging intervals.

    ``data._load_split_bed`` merges overlapping intervals, which is right for
    "does this window overlap the region set" but destroys exact-window lists
    (the 16 kb windows of this project overlap each other heavily).
    """
    import gzip

    opener = gzip.open if str(bed_path).endswith(".gz") else open
    raw: dict[str, list[tuple[int, int]]] = {}
    with opener(bed_path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if end > start:
                raw.setdefault(parts[0], []).append((start, end))
    return raw


def region_bed_mask(split_df: pd.DataFrame, indices: np.ndarray, bed_path: str,
                    mode: str = "overlap") -> tuple[np.ndarray, dict]:
    """Boolean mask over ``indices`` for the windows selected by ``bed_path``.

    ``mode='overlap'`` (default): any dataset window overlapping a BED interval,
    the same semantics as ``--test-bed`` during training.  16 kb windows overlap
    each other, so a short BED can legitimately pull in more windows than it
    lists — both counts are printed by ``select_bed_windows``.

    ``mode='exact'``: only windows whose coordinates are listed in the BED
    (0-based half-open), i.e. "the BED is a list of windows".

    Chromosome names are normalised, so a BED written as ``9`` matches data
    stored as ``chr9``.
    """
    from data import _windows_overlapping_bed

    bed = read_bed_raw(bed_path)
    if not bed:
        raise SystemExit(f"--region-bed {bed_path}: no valid intervals found")

    data_chroms = set(split_df["chr"].astype(str))
    raw: dict[str, list[tuple[int, int]]] = {}
    for chrom, intervals in bed.items():
        if chrom in data_chroms:
            key = chrom
        elif chrom.startswith("chr") and chrom[3:] in data_chroms:
            key = chrom[3:]
        elif "chr" + chrom in data_chroms:
            key = "chr" + chrom
        else:
            key = chrom
        raw.setdefault(key, []).extend(intervals)

    rows = split_df.iloc[indices]
    if mode == "exact":
        want = {(chrom, start, end)
                for chrom, intervals in raw.items() for start, end in intervals}
        hit = np.asarray([
            (str(row["chr"]), int(row["start_expanded"]) - 1, int(row["end_expanded"])) in want
            for _, row in rows.iterrows()
        ], dtype=bool)
        return hit, raw
    if mode != "overlap":
        raise SystemExit(f"unknown --region-mode {mode!r} (use overlap or exact)")

    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, intervals in raw.items():
        intervals.sort()
        out: list[list[int]] = []
        for start, end in intervals:
            if out and start <= out[-1][1]:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        arr = np.asarray(out, dtype=np.int64)
        merged[chrom] = (arr[:, 0], arr[:, 1])

    mask = _windows_overlapping_bed(
        rows["chr"].astype(str).to_numpy(),
        rows["start_expanded"].to_numpy(dtype=np.int64),
        rows["end_expanded"].to_numpy(dtype=np.int64),
        merged,
    )
    return mask, raw


def select_bed_windows(ctx, opts, split_name: str, split_indices: np.ndarray,
                       seed: int) -> np.ndarray:
    """Resolve ``--region-bed`` to the set of windows to explain."""
    composition = {}
    if opts.region_split_any:
        parts = []
        for name in ("train", "val", "test"):
            arr = np.asarray(ctx["split_idx"][name], dtype=np.int64)
            composition[name] = arr
            parts.append(arr)
        candidate = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    else:
        candidate = np.asarray(split_indices, dtype=np.int64)

    mask_overlap, bed = region_bed_mask(ctx["split_df"], candidate,
                                        opts.region_bed, mode="overlap")
    mask_exact, _ = region_bed_mask(ctx["split_df"], candidate,
                                    opts.region_bed, mode="exact")
    mask = mask_exact if opts.region_mode == "exact" else mask_overlap
    selected = candidate[mask]

    if composition:
        counts = {name: int(np.isin(selected, arr).sum())
                  for name, arr in composition.items()}
        print(f"[region] BED hits by split: {counts}")
        if counts.get("train") and not opts.allow_train_split:
            raise SystemExit(
                f"--region-bed hits {counts['train']} TRAIN windows; they are "
                f"memorised. Pass --allow-train-split to include them, or drop "
                f"--region-split-any to stay inside --split {split_name}."
            )

    n_intervals = sum(len(intervals) for intervals in bed.values())
    print(f"[region] --region-bed {opts.region_bed}: {n_intervals} intervals on "
          f"{len(bed)} chromosomes | {len(candidate)} candidate windows | "
          f"overlap={int(mask_overlap.sum())} exact={int(mask_exact.sum())} "
          f"-> using {opts.region_mode} ({selected.size} windows)")
    if selected.size == 0:
        raise SystemExit(
            f"--region-bed {opts.region_bed} selects no window of split={split_name} "
            f"(mode={opts.region_mode}).\n"
            f"  BED chromosomes : {sorted(bed)[:10]}\n"
            f"  data chromosomes: {sorted(set(ctx['split_df']['chr'].astype(str)))[:10]}\n"
            "  hint: the BED must be 0-based half-open, same assembly (mm10) and "
            "same chromosome naming (chr9 or 9 both work); if you meant to list "
            "windows, use --region-mode exact"
        )
    if opts.n_windows:
        selected = sample_window_indices(selected, int(opts.n_windows), seed)
        print(f"[region] sub-sampled to {selected.size} windows "
              f"(--n-windows {opts.n_windows})")
    return np.sort(selected)


# ---------------------------------------------------------------------------
# Command: shapley
# ---------------------------------------------------------------------------

def cmd_shapley(opts) -> int:
    t_start = time.time()
    ctx = prepare(opts)
    merged, entry, model, device = ctx["merged"], ctx["entry"], ctx["model"], ctx["device"]
    query_tracks, context_tracks = ctx["query_tracks"], ctx["context_tracks"]
    track_order, split_df = ctx["track_order"], ctx["split_df"]
    n_tracks, n_queries = len(track_order), len(query_tracks)
    n_context = len(context_tracks)

    head = opts.head
    if head not in query_tracks:
        raise SystemExit(f"--head {head} is not one of the run's query tracks {query_tracks}")
    head_idx = query_tracks.index(head)

    split = opts.split
    if split == "train" and not opts.allow_train_split:
        raise SystemExit(
            "refusing to attribute on training windows (memorised); pass "
            "--allow-train-split if you really mean it"
        )
    seed = int(merged["seed"])
    split_indices = np.asarray(ctx["split_idx"][split], dtype=np.int64)
    if opts.locus and opts.window_original_idx is not None:
        raise SystemExit("use either --locus or --window-original-idx, not both")
    single_region = bool(opts.locus) or opts.window_original_idx is not None
    if single_region and opts.region_bed:
        raise SystemExit("--region-bed cannot be combined with "
                         "--locus/--window-original-idx")
    if single_region:
        if opts.n_windows:
            raise SystemExit("--n-windows cannot be combined with a single-window selector")
        selected = select_region_windows(
            split_df, split_indices, split,
            locus=opts.locus, original_idx=opts.window_original_idx,
        )
    elif opts.region_bed:
        selected = select_bed_windows(ctx, opts, split, split_indices, seed)
    else:
        selected = sample_window_indices(split_indices, opts.n_windows, seed)
    n_windows = len(selected)
    if n_windows == 0:
        raise SystemExit(f"no windows in split={split}")

    players = resolve_players(opts.players.split(","), head, query_tracks,
                              context_tracks, model.context_specs)
    n_players = len(players)
    n_coalitions = 1 << n_players

    if opts.absence == "token":
        assert_token_absence_supported(merged, entry, players, model,
                                       query_tracks, context_tracks)

    print()
    print(f"[xai] head={head} split={split} windows={n_windows}/{len(split_indices)} | "
          f"players={n_players} coalitions={n_coalitions} absence={opts.absence}")
    if single_region:
        row = split_df.iloc[int(selected[0])]
        print(f"[region] window {row['chr']}:{int(row['start_expanded'])}-"
              f"{int(row['end_expanded'])} (original_idx={row['original_idx']})")
    if n_windows == 1:
        print("[region] single-window mode: report per-CpG attribution and the "
              "window's own share; the cluster-bootstrap CI is undefined (n/a)")
    for i, p in enumerate(players):
        role = "own trunk" if p.slot == head_idx else "context"
        print(f"      player {i}: {p.label:<12s} slot={p.slot} kind={p.kind} ({role})")
    print(f"[xai] context composition per head: "
          + "; ".join(f"{t}->{slots}" for t, slots in zip(query_tracks, model.context_specs)))
    if opts.dry_run:
        print("[xai] --dry-run: resolved everything, not touching the model")
        return 0

    # ~0.012 s per coalition-forward measured at batch 8 on one GPU
    est_min = n_windows * n_coalitions * 0.012 / 60
    print(f"[xai] rough estimate: {est_min:.0f} min on one GPU for {n_windows} windows")
    if n_windows >= 3000:
        print(f"[xai] NOTE: {n_windows} windows is a full-split-scale run "
              f"(~{est_min:.0f} min); run it in tmux/nohup for safety")

    rng_donor = np.random.default_rng(seed + 12)
    rng_seq = np.random.default_rng(seed + 13)
    rng_perm = np.random.default_rng(seed + 14)

    # ---- donor plumbing -------------------------------------------------
    pool: DonorPool | None = None
    donor_loader = None
    swap_overrides: dict[str, str] = {}
    if opts.absence in ("donor", "cross-region"):
        source = donor_source_indices(ctx, split, selected)
        pool_size = min(int(opts.donor_pool_size), len(source))
        pool_indices = sample_window_indices(source, pool_size, seed + 21)
        if len(pool_indices) == 0:
            raise SystemExit("no donor windows available (split too small)")
        pool = DonorPool.build(merged, split_df, pool_indices, track_order, n_tracks)
    if opts.absence == "cross-region":
        swap_overrides = parse_swap_paths(opts.swap_paths)
        if not swap_overrides:
            raise SystemExit("--absence cross-region needs "
                             "--swap-paths 5mc=...,atac=...,rna=...")
        _, donor_loader = make_loader(merged, split_df, selected, track_order,
                                      opts.batch_size, opts.num_workers, swap_overrides)
        print("[xai] cross-region donors: "
              + ", ".join(f"{k}={v}" for k, v in swap_overrides.items())
              + (" | other players use the random-donor pool"
                 if len(swap_overrides) < len(track_order) else ""))
    elif opts.absence == "mean":
        pool_indices = sample_window_indices(split_indices, int(opts.donor_pool_size),
                                             seed + 21)
        pool = DonorPool.build(merged, split_df, pool_indices, track_order, n_tracks)

    # ---- main loop ------------------------------------------------------
    _, loader = make_loader(merged, split_df, selected, track_order,
                            opts.batch_size, opts.num_workers)
    donor_iter = iter(donor_loader) if donor_loader is not None else None

    base_seed = seed + 2000
    mask_fraction = float(merged["mask_fraction"])
    obs_rows: list[dict] = []
    layer_buf: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    flat = {"phi": {}, "inter": {}, "fact": {}, "div": {}}
    per_window = {"phi": {}}
    per_window_extra: dict[str, list[float]] = {}
    div_sums: dict[int, float] = {}
    coalition_buf: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    counts: list[int] = []
    efficiency_err = 0.0
    dividend_err = 0.0
    sum_y = sum_y2 = 0.0
    sse = 0.0
    n_values = 0
    g = 0
    batch_index = 0

    for batch in loader:
        tracks, sequence, base_ids = split_batch(batch, n_tracks)
        base_ids_cpu = base_ids  # CPU copy: generate_pretraining_cpg_mask needs numpy
        tracks = [t.to(device) for t in tracks]
        sequence, base_ids = sequence.to(device), base_ids.to(device)
        bsz, seq_len = base_ids.shape
        if seq_len != int(merged["target_length"]):
            raise SystemExit(
                f"window length {seq_len} != {merged['target_length']}: the model is "
                f"length-locked and cannot score a padded/truncated window"
            )

        position_mask = torch.stack([
            generate_pretraining_cpg_mask(base_ids_cpu[i:i + 1], mask_fraction,
                                          seed=base_seed + g + i)[0]
            for i in range(bsz)
        ], dim=0).to(device)
        scored = scored_positions(base_ids, position_mask, opts.positions)

        donor_tracks = None
        if opts.absence in ("donor", "cross-region"):
            ids = rng_donor.integers(0, pool.size, size=bsz)
            if pool.indices is not None:
                targets = selected[g:g + bsz]
                for b in range(bsz):
                    if pool.indices[ids[b]] == targets[b]:
                        ids[b] = (ids[b] + 1) % pool.size
            donor_tracks = {name: pool.gather(name, ids, device) for name in track_order}
        if opts.absence == "cross-region":
            d_tracks, _d_seq, d_base = split_batch(next(donor_iter), n_tracks)
            d_tracks = [t.to(device) for t in d_tracks]
            if not torch.equal(d_base.to(device), base_ids):
                raise SystemExit("cross-region loader is misaligned with the attribution loader")
            for name, tensor in zip(track_order, d_tracks):
                if name in swap_overrides:
                    donor_tracks[name] = tensor

        absent = prepare_absences(players, tracks, sequence, base_ids, opts.absence,
                                  donor_tracks, pool, rng_seq, rng_perm,
                                  opts.seq_shuffle, device)
        presence = None
        if opts.absence == "token":
            ones = lambda: torch.ones(bsz, 1, 1, device=device)  # noqa: E731
            presence = {
                "query": [ones() for _ in range(n_queries)],
                "track": [ones() for _ in range(n_context)],
                "seq": ones(),
            }

        values = run_coalitions(model, players, absent, tracks, sequence,
                                position_mask, head_idx, n_queries, n_context,
                                opts.absence, presence)
        full = n_coalitions - 1
        phi = shapley_values(values, n_players)
        inter = pairwise_shapley_interaction(values, n_players)
        fact = factorial_interaction(values, n_players)
        div = harsanyi_dividends(values, n_players)

        # hard invariants, evaluated on scored positions only
        if scored.any():
            total = (values[full] - values[0])[scored]
            phi_sum = sum(t[scored] for t in phi)
            efficiency_err = max(efficiency_err,
                                 float((phi_sum - total).abs().max().item()))
            div_sum = sum(div[s][scored] for s in range(1, n_coalitions))
            dividend_err = max(dividend_err,
                               float((div_sum - total).abs().max().item()))
            target = tracks[head_idx].squeeze(-1)[scored].double()
            pred = values[full][scored].double()
            sse += float(((pred - target) ** 2).sum().item())
            sum_y += float(target.sum().item())
            sum_y2 += float((target ** 2).sum().item())
            n_values += int(target.numel())
            # exact decomposition needs every order, not just the exported |S|<=2
            for subset, tensor in div.items():
                if subset == 0:
                    continue
                div_sums[subset] = div_sums.get(subset, 0.0) + float(
                    tensor[scored].sum().item()
                )

        # per-window accumulation: every name gets exactly one entry per window
        scored_np = scored.cpu().numpy()
        for i in range(bsz):
            meta = split_df.iloc[int(selected[g + i])]
            pos = np.flatnonzero(scored_np[i])
            obs_rows.append({
                "chr": str(meta["chr"]),
                "start_expanded": int(meta["start_expanded"]),
                "end_expanded": int(meta["end_expanded"]),
                "original_idx": int(meta["original_idx"]) if "original_idx" in meta else -1,
                "n_scored": int(pos.size),
            })
            counts.append(int(pos.size))
            if pos.size:
                pred_vals = values[full][i].detach().cpu().numpy()[pos]
                target_vals = tracks[head_idx][i].detach().cpu().numpy()[pos].reshape(-1)
                per_window_extra.setdefault("pred", []).append(float(pred_vals.mean()))
                per_window_extra.setdefault("target", []).append(float(target_vals.mean()))
            else:
                per_window_extra.setdefault("pred", []).append(float("nan"))
                per_window_extra.setdefault("target", []).append(float("nan"))
            for (a, b), tensor in inter.items():
                key = f"{players[a].name}__{players[b].name}"
                inter_vals = tensor[i].detach().cpu().numpy()[pos] if pos.size else np.zeros(0)
                per_window_extra.setdefault(f"inter_{key}", []).append(float(inter_vals.sum()))
            row_ids = np.full(pos.size, g + i, dtype=np.int64)
            for a, player in enumerate(players):
                vals = phi[a][i].detach().cpu().numpy()[pos] if pos.size else np.zeros(0, np.float32)
                append(flat["phi"], player.name, vals.astype(np.float32))
                per_window["phi"].setdefault(player.name, []).append(float(vals.sum()))
            for (a, b), tensor in inter.items():
                key = f"{players[a].name}__{players[b].name}"
                vals = tensor[i].detach().cpu().numpy()[pos] if pos.size else np.zeros(0, np.float32)
                append(flat["inter"], key, vals.astype(np.float32))
            for (a, b), tensor in fact.items():
                key = f"{players[a].name}__{players[b].name}"
                vals = tensor[i].detach().cpu().numpy()[pos] if pos.size else np.zeros(0, np.float32)
                append(flat["fact"], key, vals.astype(np.float32))
            for subset, tensor in div.items():
                if subset == 0:
                    continue
                if subset.bit_count() > 2 and not opts.save_all_dividends:
                    continue
                key = subset_name(players, subset)
                vals = tensor[i].detach().cpu().numpy()[pos] if pos.size else np.zeros(0, np.float32)
                append(flat["div"], key, vals.astype(np.float32))
            if pos.size:
                for a, player in enumerate(players):
                    push(layer_buf, f"phi_{player.name}", row_ids, pos,
                         phi[a][i].detach().cpu().numpy()[pos])
                for (a, b), tensor in inter.items():
                    push(layer_buf, f"inter_{players[a].name}__{players[b].name}",
                         row_ids, pos, tensor[i].detach().cpu().numpy()[pos])
                for (a, b), tensor in fact.items():
                    push(layer_buf, f"fact_{players[a].name}__{players[b].name}",
                         row_ids, pos, tensor[i].detach().cpu().numpy()[pos])
                for subset, tensor in div.items():
                    if subset == 0 or (subset.bit_count() > 2 and not opts.save_all_dividends):
                        continue
                    push(layer_buf, f"div_{subset_name(players, subset)}", row_ids, pos,
                         tensor[i].detach().cpu().numpy()[pos])
                push(layer_buf, "pred", row_ids, pos,
                     values[full][i].detach().cpu().numpy()[pos])
                push(layer_buf, "target", row_ids, pos,
                     tracks[head_idx][i].detach().cpu().numpy()[pos])
                push(layer_buf, "mask", row_ids, pos, np.ones(pos.size, dtype=np.float32))
                push(layer_buf, "base_encoded", row_ids, pos,
                     base_ids[i].detach().cpu().numpy()[pos].astype(np.float32))
                push(layer_buf, "motif", row_ids, pos, motif_codes(base_ids[i], pos))
                if opts.save_coalition_values:
                    for coalition, tensor in values.items():
                        push(coalition_buf, str(coalition), row_ids, pos,
                             tensor[i].detach().cpu().numpy()[pos])

        g += bsz
        batch_index += 1
        if batch_index % 10 == 0 or g >= n_windows:
            print(f"  [{g}/{n_windows}] efficiency_err={efficiency_err:.2e} "
                  f"dividend_err={dividend_err:.2e} ({time.time() - t_start:.0f}s)")

    # ---- invariants ------------------------------------------------------
    tol = float(opts.tolerance_math)
    print()
    print(f"[check] Shapley efficiency   max|sum(phi) - (v(full)-v(empty))| = {efficiency_err:.3e}")
    print(f"[check] dividend identity    max|sum(m)  - (v(full)-v(empty))| = {dividend_err:.3e}")
    if efficiency_err > tol or dividend_err > tol:
        raise SystemExit(f"invariant violated (tolerance {tol:g}) — refusing to write results")
    ss_tot = sum_y2 - (sum_y ** 2) / max(1, n_values)
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    print(f"[check] reconstruction R2 on the scored positions of this subset = {r2:.4f} "
          f"({n_values} CpGs)")

    # ---- random-init control --------------------------------------------
    control = None
    if opts.control == "random-init":
        if opts.absence in ("token", "cross-region"):
            print(f"\n[control] skipped: not defined for absence={opts.absence}",
                  file=sys.stderr)
        else:
            control = run_random_init_control(ctx, players, opts, selected, base_seed,
                                              mask_fraction, rng_donor, n_tracks,
                                              head_idx, n_queries, n_context)

    # ---- outputs ---------------------------------------------------------
    out_dir = Path(opts.out_dir) if opts.out_dir else (
        Path(merged["_json_path"]).parent / "xai"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    stem = (f"{ts}_modality_shapley_{head}_{split}_n{n_windows}"
            + (f"_{opts.tag}" if opts.tag else ""))

    summary = summarise(players, flat, per_window, counts, int(opts.bootstrap), seed)
    summary_path = out_dir / f"{stem}_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"[out] {summary_path}")

    decomp = decomposition_table(players, div_sums, n_values, n_players)
    decomp_path = out_dir / f"{stem}_decomposition.csv"
    pd.DataFrame(decomp).to_csv(decomp_path, index=False)
    print(f"[out] {decomp_path}")

    strata_path = None
    if not opts.no_strata:
        strata = strata_table(merged, players, flat, obs_rows, counts, split_df, selected)
        if strata:
            strata_path = out_dir / f"{stem}_strata.csv"
            pd.DataFrame(strata).to_csv(strata_path, index=False)
            print(f"[out] {strata_path}")

    per_window_path = out_dir / f"{stem}_per_window.csv"
    pd.DataFrame(per_window_table(players, per_window, per_window_extra,
                                  counts, obs_rows)).to_csv(per_window_path, index=False)
    print(f"[out] {per_window_path}")

    h5ad_path = ""
    if not opts.no_h5ad:
        h5ad_path = str(out_dir / f"{stem}.h5ad")
        try:
            write_h5ad(h5ad_path, obs_rows, layer_buf, players, merged, opts,
                       head, split, efficiency_err, dividend_err, r2)
            print(f"[out] {h5ad_path}")
        except Exception as exc:  # never lose a finished run on an optional export
            import traceback
            print(f"[out] WARNING: h5ad export failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            h5ad_path = ""

    if opts.save_coalition_values and coalition_buf:
        npz_path = out_dir / f"{stem}_coalition_values.npz"
        payload = {"shape": np.array([n_windows, int(merged["target_length"])])}
        for coalition, chunks in coalition_buf.items():
            payload[f"v{coalition}_rows"] = np.concatenate([c[0] for c in chunks])
            payload[f"v{coalition}_cols"] = np.concatenate([c[1] for c in chunks])
            payload[f"v{coalition}_vals"] = np.concatenate([c[2] for c in chunks])
        np.savez_compressed(npz_path, **payload)
        print(f"[out] {npz_path}")

    manifest = {
        "version": VERSION,
        "command": sys.argv,
        "git_commit": git_commit(),
        "checkpoint": merged["_checkpoint_path"],
        "checkpoint_sha256": sha256_of_file(merged["_checkpoint_path"]),
        "results_json": merged["_json_path"],
        "head": head,
        "split": split,
        "n_windows": n_windows,
        "split_windows": int(len(split_indices)),
        "players": [{"name": p.name, "slot": p.slot, "kind": p.kind, "label": p.label}
                    for p in players],
        "n_coalitions": n_coalitions,
        "absence": opts.absence,
        "seq_shuffle": opts.seq_shuffle,
        "donor_pool_size": int(opts.donor_pool_size),
        "positions": opts.positions,
        "mask_fraction": mask_fraction,
        "mask_seed_base": base_seed,
        "seed": seed,
        "efficiency_max_abs_err": efficiency_err,
        "dividend_max_abs_err": dividend_err,
        "subset_reconstruction_r2": r2,
        "n_scored_values": n_values,
        "random_init_control": control,
        "region": {
            "locus": opts.locus,
            "original_idx": opts.window_original_idx,
            "bed": opts.region_bed,
            "bed_mode": opts.region_mode,
            "split_any": bool(opts.region_split_any),
            "single_window": bool(n_windows == 1),
        },
        "outputs": {
            "h5ad": h5ad_path,
            "summary_csv": str(summary_path),
            "decomposition_csv": str(decomp_path),
            "strata_csv": str(strata_path) if strata_path else "",
            "per_window_csv": str(per_window_path),
        },
        "runtime_s": round(time.time() - t_start, 1),
    }
    manifest_path = out_dir / f"{stem}_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[out] {manifest_path}")

    print_summary(players, summary, decomp)
    print(f"\n[xai] done in {time.time() - t_start:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# Token-absence guard / path parsing
# ---------------------------------------------------------------------------

def assert_token_absence_supported(merged, entry, players, model, query_tracks,
                                   context_tracks):
    """Fail loudly when the modality-mask tokens were never trained."""
    drop = entry.get("modality_drop_probs") or {}
    sd = model.state_dict()
    problems = []
    for p in players:
        if p.kind == "seq":
            p_drop = float(merged.get("drop_p_seq", 0.0) or 0.0)
            token = sd.get("sequence_mask_token")
        elif p.slot < len(query_tracks):
            p_drop = float(drop.get(p.name, merged.get(f"drop_p_{p.name}", 0.0)) or 0.0)
            token = sd.get(f"query_mask_tokens.{p.slot}")
        else:
            p_drop = float(drop.get(p.name, merged.get(f"drop_p_{p.name}", 0.0)) or 0.0)
            token = sd.get(f"context_track_mask_tokens.{p.slot - len(query_tracks)}")
        norm = float(token.abs().max().item()) if token is not None else 0.0
        if p_drop <= 0.0 or norm == 0.0:
            problems.append(f"  {p.label}: drop_p={p_drop:g}, |token|max={norm:.3g}")
    if problems:
        raise SystemExit(
            "--absence token is not usable on this checkpoint: the modality-mask "
            "tokens were never trained (zero-initialised and/or drop_p=0), so "
            "'absent' would substitute a zero vector instead of a learned state.\n"
            + "\n".join(problems)
            + "\n  use --absence donor / permute / cross-region / mean / zero instead."
        )


def parse_swap_paths(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    out = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise SystemExit(f"--swap-paths entry {item!r} must look like name=path")
        name, path = item.split("=", 1)
        name = name.strip()
        if name not in TRACK_INDEX:
            raise SystemExit(f"--swap-paths: unknown track {name!r}")
        out[name] = path.strip()
    return out


def subset_name(players: list[Player], subset: int) -> str:
    return "+".join(players[i].name for i in range(len(players)) if (subset >> i) & 1)


def push(store: dict, key: str, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray):
    # (L, 1) track slices arrive 2-D; the COO constructor needs 1-D everywhere.
    store.setdefault(key, []).append((
        np.asarray(rows, dtype=np.int64).reshape(-1),
        np.asarray(cols, dtype=np.int64).reshape(-1),
        np.asarray(vals, dtype=np.float32).reshape(-1),
    ))


def motif_codes(base_ids: torch.Tensor, pos: np.ndarray) -> np.ndarray:
    """1 = CpG C/G, 2 = other C, 0 = everything else (matches utils convention)."""
    ids = base_ids.detach().cpu().numpy().astype(np.int64)
    left = np.concatenate([[0], ids[:-1]])
    right = np.concatenate([ids[1:], [0]])
    cpg = ((ids == 1) & (right == 2)) | ((ids == 2) & (left == 1))
    other_c = (ids == 1) & ~cpg
    codes = np.zeros(ids.size, dtype=np.float32)
    codes[cpg] = 1.0
    codes[other_c] = 2.0
    return codes[pos]


# ---------------------------------------------------------------------------
# Random-initialised control
# ---------------------------------------------------------------------------

def run_random_init_control(ctx, players, opts, selected, base_seed, mask_fraction,
                            rng_donor, n_tracks, head_idx, n_queries, n_context):
    """Same pipeline on a randomly initialised model; the attribution *profile*
    must depend on the weights (|phi| rank correlations are reported but are
    confounded by the shared local signal scale)."""
    print("\n[control] random-initialised model (same architecture, fresh weights)")
    merged, split_df = ctx["merged"], ctx["split_df"]
    device = ctx["device"]
    n = min(int(opts.control_windows), len(selected))
    sub = selected[:n]

    torch.manual_seed(int(merged["seed"]) + 1)
    torch.cuda.manual_seed_all(int(merged["seed"]) + 1)
    random_model = MultiQueryMaskedPretrainingModel(
        seq_len=int(merged["target_length"]),
        hidden_dim=int(merged["hidden_dim"]),
        use_positional_encoding=bool(merged.get("use_positional_encoding", False)),
        num_blocks=int(merged["num_blocks"]),
        fusion_type=str(merged["fusion_type"]),
        num_queries=len(ctx["query_tracks"]),
        num_context_tracks=len(ctx["context_tracks"]),
        context_mode=str(merged["context_mode"]),
    ).to(device).eval().float()
    for param in random_model.parameters():
        param.requires_grad_(False)

    pool = None
    if opts.absence in ("donor", "mean"):
        source = donor_source_indices(ctx, opts.split, sub)
        pool_idx = sample_window_indices(source, 128, int(merged["seed"]) + 34)
        if len(pool_idx):
            pool = DonorPool.build(merged, split_df, pool_idx, ctx["track_order"], n_tracks)

    _, loader = make_loader(merged, split_df, sub, ctx["track_order"],
                            opts.batch_size, 0)
    rng_seq = np.random.default_rng(int(merged["seed"]) + 41)
    rng_perm = np.random.default_rng(int(merged["seed"]) + 42)

    real_phi = {p.name: [] for p in players}
    rand_phi = {p.name: [] for p in players}
    g = 0
    for batch in loader:
        tracks, sequence, base_ids = split_batch(batch, n_tracks)
        base_ids_cpu = base_ids
        tracks = [t.to(device) for t in tracks]
        sequence, base_ids = sequence.to(device), base_ids.to(device)
        bsz = base_ids.shape[0]
        position_mask = torch.stack([
            generate_pretraining_cpg_mask(base_ids_cpu[i:i + 1], mask_fraction,
                                          seed=base_seed + g + i)[0]
            for i in range(bsz)
        ], dim=0).to(device)
        scored = scored_positions(base_ids, position_mask, opts.positions).cpu().numpy()

        donor_tracks = None
        if opts.absence == "donor":
            if pool is None:
                print("[control] no disjoint donor pool; skipping the control",
                      file=sys.stderr)
                return None
            ids = rng_donor.integers(0, pool.size, size=bsz)
            donor_tracks = {name: pool.gather(name, ids, device) for name in ctx["track_order"]}

        # ONE absence realisation per batch, shared by both models, so the
        # comparison isolates the weights and not the intervention draw.
        absent = prepare_absences(players, tracks, sequence, base_ids, opts.absence,
                                  donor_tracks, pool, rng_seq, rng_perm,
                                  opts.seq_shuffle, device)
        for model_obj, sink in ((ctx["model"], real_phi), (random_model, rand_phi)):
            values = run_coalitions(model_obj, players, absent, tracks, sequence,
                                    position_mask, head_idx, n_queries, n_context,
                                    opts.absence, None)
            phi = shapley_values(values, len(players))
            for i in range(bsz):
                pos = np.flatnonzero(scored[i])
                if pos.size == 0:
                    continue
                for a, p in enumerate(players):
                    sink[p.name].append(
                        phi[a][i].detach().cpu().numpy()[pos].astype(np.float32)
                    )
        g += bsz

    # Between-window differences in signal scale make a *pooled* |phi| correlation
    # uninformative (a random model also produces big effects where the input
    # variance is big).  The informative statistic is the within-window rank
    # correlation, which removes that shared scale factor, plus the share profile.
    within = {}
    shares = {"trained": {}, "random": {}}
    for p in players:
        if not real_phi[p.name] or not rand_phi[p.name]:
            continue
        per_window = []
        for a_w, b_w in zip(real_phi[p.name], rand_phi[p.name]):
            if a_w.size >= 5 and a_w.size == b_w.size:
                rho = spearman(np.abs(a_w), np.abs(b_w))
                if np.isfinite(rho):
                    per_window.append(rho)
        if per_window:
            within[p.label] = float(np.mean(per_window))
        real_abs = float(np.mean([np.abs(x).mean() for x in real_phi[p.name] if x.size]))
        rand_abs = float(np.mean([np.abs(x).mean() for x in rand_phi[p.name] if x.size]))
        shares["trained"][p.label] = real_abs
        shares["random"][p.label] = rand_abs
    for key in shares:
        total = sum(shares[key].values())
        shares[key] = {k: (v / total if total else float("nan")) for k, v in shares[key].items()}

    finite = [abs(v) for v in within.values() if np.isfinite(v)]
    max_abs = max(finite) if finite else float("nan")
    n_tested = sum(int(np.concatenate(real_phi[p.name]).size)
                   for p in players if real_phi[p.name])
    # Verdict: does the *profile* of attribution mass depend on the weights?
    # Raw |phi| rank correlations are dominated by the shared local signal scale,
    # so they are reported but not used to pass/fail (a random model still gets
    # big |phi| where the input variance is big).
    share_l1 = 0.5 * sum(abs(shares["trained"].get(k, 0.0) - shares["random"].get(k, 0.0))
                         for k in set(shares["trained"]) | set(shares["random"]))
    if n_tested < 200:
        status = "INSUFFICIENT"
    elif share_l1 >= 0.15:
        status = "PASS"
    else:
        status = "WARN"
    print("[control] within-window rank corr(|phi| trained, |phi| random): "
          + ", ".join(f"{k}={v:+.3f}" for k, v in within.items()) + "  [diagnostic only]")
    print("[control] |phi| share trained: "
          + ", ".join(f"{k}={v*100:.0f}%" for k, v in shares["trained"].items())
          + " | random: "
          + ", ".join(f"{k}={v*100:.0f}%" for k, v in shares["random"].items())
          + f" | L1={share_l1:.2f} [{status}]")
    return {"within_window_spearman_abs_phi": within,
            "max_abs_within_spearman": max_abs,
            "share_abs_trained": shares["trained"],
            "share_abs_random": shares["random"],
            "share_l1_distance": share_l1,
            "n_windows": int(n), "n_values": int(n_tested), "status": status}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise(players, flat, per_window, counts, n_boot, seed) -> list[dict]:
    phi_stack = np.stack([
        np.concatenate(flat["phi"][p.name]) if flat["phi"].get(p.name) else np.zeros(0)
        for p in players
    ]) if flat["phi"] else np.zeros((len(players), 0))
    abs_means = np.array([float(np.abs(phi_stack[i]).mean()) for i in range(len(players))])
    share = abs_means / abs_means.sum() if abs_means.sum() > 0 else abs_means * np.nan
    win_counts = np.asarray(counts, dtype=np.float64)

    rows = []
    for i, p in enumerate(players):
        win_sums = np.asarray(per_window["phi"].get(p.name, []), dtype=np.float64)
        point, lo, hi = cluster_bootstrap_mean(win_sums, win_counts, n_boot, seed + i)
        vals = phi_stack[i]
        rows.append({
            "kind": "player",
            "name": p.name,
            "label": p.label,
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "mean_window_weighted": point,
            "mean_abs": float(abs_means[i]),
            "share_abs": float(share[i]),
            "frac_positive": float((vals > 0).mean()) if vals.size else float("nan"),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_values": int(vals.size),
        })

    for (a, b) in combinations(range(len(players)), 2):
        key = f"{players[a].name}__{players[b].name}"
        inter = np.concatenate(flat["inter"][key]) if flat["inter"].get(key) else np.zeros(0)
        fact = np.concatenate(flat["fact"][key]) if flat["fact"].get(key) else np.zeros(0)
        rows.append({
            "kind": "interaction",
            "name": f"{players[a].label} x {players[b].label}",
            "label": key,
            "mean": float(inter.mean()) if inter.size else float("nan"),
            "mean_abs": float(np.abs(inter).mean()) if inter.size else float("nan"),
            "share_abs": float("nan"),
            "frac_positive": float((inter > 0).mean()) if inter.size else float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_values": int(inter.size),
            "mean_factorial": float(fact.mean()) if fact.size else float("nan"),
        })
    return rows


def decomposition_table(players, div_sums: dict[int, float], n_values: int,
                        n_players: int) -> list[dict]:
    """Exact additive decomposition; div_sums holds the summed m(S) over CpGs."""
    if not div_sums or n_values == 0:
        return []
    means = {subset: total / n_values for subset, total in div_sums.items()}
    total = sum(means.values())
    abs_total = sum(abs(v) for v in means.values())
    rows = []
    for subset, mean in sorted(means.items(), key=lambda kv: (kv[0].bit_count(), kv[0])):
        rows.append({
            "subset": subset_name(players, subset),
            "order": subset.bit_count(),
            "mean_dividend": mean,
            "share_of_total_signed": mean / total if total else float("nan"),
            "share_of_abs_total": abs(mean) / abs_total if abs_total else float("nan"),
        })
    rows.append({"subset": "__total__", "order": -1, "mean_dividend": total,
                 "share_of_total_signed": 1.0, "share_of_abs_total": 1.0})
    return rows


def per_window_table(players, per_window, per_window_extra, counts,
                     obs_rows) -> list[dict]:
    """One row per window: its own phi / share / interaction (no cross-window mix).

    This is what a "score for this region" should be read from: a single 16 kb
    window's composition, not the global average.
    """
    rows = []
    for w, obs in enumerate(obs_rows):
        n = int(counts[w]) if w < len(counts) else 0
        row = dict(obs)
        row["pred_mean"] = extra_at(per_window_extra, "pred", w)
        row["target_mean"] = extra_at(per_window_extra, "target", w)
        means = {}
        for p in players:
            seq = per_window["phi"].get(p.name)
            total = float(seq[w]) if seq and w < len(seq) else float("nan")
            means[p.name] = total / n if n else float("nan")
        abs_total = sum(abs(v) for v in means.values() if np.isfinite(v))
        for p in players:
            row[f"phi_{p.name}_mean"] = means[p.name]
            row[f"phi_{p.name}_share"] = (
                abs(means[p.name]) / abs_total if abs_total else float("nan")
            )
        for a, b in combinations(range(len(players)), 2):
            key = f"inter_{players[a].name}__{players[b].name}"
            row[key + "_mean"] = (
                extra_at(per_window_extra, key, w) / n if n else float("nan")
            )
        rows.append(row)
    return rows


def extra_at(store: dict[str, list[float]], key: str, index: int) -> float:
    seq = store.get(key)
    return float(seq[index]) if seq and index < len(seq) else float("nan")


def strata_table(merged, players, flat, obs_rows, counts, split_df, selected):
    """Mean phi by chromosome and by cCRE class (4th column of the DMR BED)."""
    if not flat.get("phi"):
        return []
    order = [p.name for p in players]
    stacks = []
    for name in order:
        chunks = flat["phi"].get(name) or []
        stacks.append(np.concatenate(chunks) if chunks else np.zeros(0))
    matrix = np.stack(stacks)
    window_ids = np.repeat(np.arange(len(counts), dtype=np.int64),
                           np.asarray(counts, dtype=np.int64))
    if window_ids.size != matrix.shape[1]:
        window_ids = window_ids[:matrix.shape[1]]

    chr_col = np.asarray([row["chr"] for row in obs_rows])
    ccre_col = ccre_classes(merged, selected, split_df)
    rows = []
    for kind, values in (("chr", chr_col), ("ccre", ccre_col)):
        if values is None:
            continue
        for level in pd.unique(values):
            window_mask = np.asarray(values == level)
            if window_mask.sum() == 0:
                continue
            value_mask = np.isin(window_ids, np.flatnonzero(window_mask))
            if value_mask.sum() == 0:
                continue
            for i, player in enumerate(players):
                vals = matrix[i][value_mask]
                rows.append({
                    "stratum_type": kind,
                    "stratum": str(level),
                    "player": player.name,
                    "n_windows": int(window_mask.sum()),
                    "n_values": int(vals.size),
                    "mean": float(vals.mean()),
                    "mean_abs": float(np.abs(vals).mean()),
                })
    return rows


def ccre_classes(merged, selected, split_df):
    """Region class from the 4th column of the DMR BED (e.g. 'CpGisland_5')."""
    path = first_path(merged.get("dmr_csv"))
    if not path or not str(path).lower().endswith((".bed", ".bed.gz")):
        return None
    try:
        names = pd.read_csv(path, sep="\t", header=None, usecols=[3],
                            names=["name"], comment="#")["name"].astype(str)
    except Exception:
        return None
    classes = names.str.split("_").str[0].to_numpy()
    out = []
    for idx in selected:
        meta = split_df.iloc[int(idx)]
        row_idx = int(meta["original_idx"]) if "original_idx" in meta else int(idx)
        out.append(classes[row_idx] if 0 <= row_idx < len(classes) else "unknown")
    return np.asarray(out)


def print_summary(players, summary, decomp):
    print()
    print("=" * 82)
    print("MODALITY SHAPLEY  (units = target units; for these runs 5hmC %, 0-100;")
    print("                   positive pushes the prediction up)")
    print("=" * 82)
    rows = []
    for row in summary:
        if row["kind"] != "player":
            continue
        ci = ("n/a" if not np.isfinite(row["ci_lo"])
              else f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]")
        rows.append([
            row["label"], f"{row['mean']:+.4f}", f"{row['mean_abs']:.4f}",
            f"{row['share_abs'] * 100:.1f}%",
            f"{row['frac_positive'] * 100:.0f}%",
            ci,
        ])
    print(fmt_table(rows, ["player", "mean phi", "mean |phi|", "share", "phi>0", "95% CI"]))
    print()
    inter_rows = [[
        row["name"], f"{row['mean']:+.5f}", f"{row['mean_factorial']:+.5f}",
        f"{row['frac_positive'] * 100:.0f}%",
    ] for row in summary if row["kind"] == "interaction"]
    if inter_rows:
        print("PAIRWISE INTERACTION  (>0 synergy, <0 redundancy; not additive with phi)")
        print(fmt_table(inter_rows, ["pair", "Shapley I", "2x2 factorial", "I>0"]))
        print()
    if decomp:
        total = next(r for r in decomp if r["subset"] == "__total__")
        rows = [[
            r["subset"], str(r["order"]), f"{r['mean_dividend']:+.5f}",
            f"{r['share_of_total_signed'] * 100:+.1f}%"
            if np.isfinite(r["share_of_total_signed"]) else "n/a",
        ] for r in decomp if r["subset"] != "__total__"]
        print(f"HARSANYI DECOMPOSITION  (exact: v(P)-v(empty) = "
              f"{total['mean_dividend']:+.5f})")
        print(fmt_table(rows, ["subset", "order", "mean m(S)", "share signed"]))


# ---------------------------------------------------------------------------
# h5ad export
# ---------------------------------------------------------------------------

def write_h5ad(path, obs_rows, layer_buf, players, merged, opts, head, split,
               efficiency_err, dividend_err, r2):
    import anndata as ad
    from scipy.sparse import coo_matrix

    n_obs = len(obs_rows)
    seq_len = int(merged["target_length"])
    obs = pd.DataFrame(obs_rows)
    obs.index = obs.index.astype(str)
    var = pd.DataFrame({"position_idx": np.arange(seq_len)})
    layers = {}
    for name, chunks in layer_buf.items():
        rows = np.concatenate([c[0] for c in chunks])
        cols = np.concatenate([c[1] for c in chunks])
        vals = np.concatenate([c[2] for c in chunks])
        layers[name] = coo_matrix((vals, (rows, cols)), shape=(n_obs, seq_len)).tocsc()

    x_key = f"phi_{head}"
    matrix = layers[x_key] if x_key in layers else next(iter(layers.values()))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    for name, mat in layers.items():
        adata.layers[name] = mat
    adata.uns.update({
        "version": VERSION,
        "source": "run_xai_modality_shapley.py",
        "checkpoint": merged["_checkpoint_path"],
        "results_json": merged["_json_path"],
        "head": head,
        "split": split,
        "players": [p.label for p in players],
        "absence": opts.absence,
        "seq_shuffle": opts.seq_shuffle,
        "positions": opts.positions,
        "mask_fraction": float(merged["mask_fraction"]),
        "seed": int(merged["seed"]),
        "efficiency_max_abs_err": float(efficiency_err),
        "dividend_max_abs_err": float(dividend_err),
        "subset_reconstruction_r2": float(r2),
    })
    ensure_parent_dir(path)
    adata.write_h5ad(path, compression="gzip")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact modality-level Shapley / interaction analysis for "
                    "MultiQueryMaskedPretrainingModel checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True,
                        help="path to *_best_*.pt written by run_multiquery_pretraining.py")
    common.add_argument("--results-json", required=True,
                        help="path to the matching *_results.json")
    common.add_argument("--device", default="auto", help="'auto', 'cpu', or e.g. 'cuda:5'")
    common.add_argument("--seed", type=int, default=None,
                        help="override the checkpoint's random seed")
    common.add_argument("--mask-fraction", type=float, default=None,
                        help="override the checkpoint's position-mask fraction")
    common.add_argument("--test-regions-bed", default=None,
                        help="optional *_test_regions.bed for an exact split check")
    common.add_argument("--out-dir", default=None,
                        help="default: <results-json dir>/xai")
    common.add_argument("--num-workers", type=int, default=2)

    verify = sub.add_parser("verify", parents=[common],
                            help="reproduce the reported metrics (proves the rebuild)")
    verify.add_argument("--split", default="test", choices=["test", "val"])
    verify.add_argument("--batch-size", type=int, default=32)
    verify.add_argument("--tolerance", type=float, default=2e-3,
                        help="allowed |rebuilt R2 - JSON R2|")
    verify.set_defaults(func=cmd_verify)

    shap = sub.add_parser("shapley", parents=[common],
                          help="exact Shapley + interaction indices (2^n forwards/window)")
    shap.add_argument("--head", default="5hmc", help="which query head to explain")
    shap.add_argument("--split", default="test", choices=["test", "val", "train"])
    shap.add_argument("--n-windows", type=int, default=0, help="0 = every window")
    shap.add_argument("--locus", default=None,
                      help="explain one window, e.g. chr9:119592828-119609211 "
                           "(1-based, comma-free); mutually exclusive with --n-windows")
    shap.add_argument("--window-original-idx", type=int, default=None,
                      help="explain one window by its original_idx (column 4 of "
                           "*_test_regions.bed); mutually exclusive with --locus")
    shap.add_argument("--region-bed", default=None,
                      help="explain only windows overlapping this BED (0-based "
                           "half-open, mm10); an omitted --n-windows means all hits")
    shap.add_argument("--region-split-any", action="store_true",
                      help="with --region-bed: search train/val/test instead of only "
                           "--split (train hits still need --allow-train-split)")
    shap.add_argument("--region-mode", default="overlap", choices=["overlap", "exact"],
                      help="with --region-bed: 'overlap' = any window touching the "
                           "BED (training --test-bed semantics); 'exact' = only "
                           "windows whose coordinates are listed")
    shap.add_argument("--players", default="seq,other,atac,rna,self",
                      help="comma list of seq/other/self/<track>; 4 players -> 16 forwards")
    shap.add_argument("--absence", default="donor",
                      choices=["donor", "permute", "cross-region", "mean", "zero", "token"])
    shap.add_argument("--seq-shuffle", default="at", choices=["at", "cpg"])
    shap.add_argument("--donor-pool-size", type=int, default=256)
    shap.add_argument("--swap-paths", default=None,
                      help="cross-region donors: '5mc=...,atac=...,rna=...'")
    shap.add_argument("--positions", default="cpg_forward", choices=["cpg_forward", "cpg", "c"])
    shap.add_argument("--batch-size", type=int, default=8)
    shap.add_argument("--bootstrap", type=int, default=1000)
    shap.add_argument("--control", default="random-init", choices=["none", "random-init"])
    shap.add_argument("--control-windows", type=int, default=25)
    shap.add_argument("--tolerance-math", type=float, default=5e-4,
                      help="max allowed invariant violation")
    shap.add_argument("--save-coalition-values", action="store_true")
    shap.add_argument("--save-all-dividends", action="store_true",
                      help="also export |S|>=3 dividends (default: |S|<=2 only)")
    shap.add_argument("--no-h5ad", action="store_true")
    shap.add_argument("--no-strata", action="store_true")
    shap.add_argument("--allow-train-split", action="store_true")
    shap.add_argument("--tag", default="")
    shap.add_argument("--dry-run", action="store_true")
    shap.set_defaults(func=cmd_shapley)
    return parser


def main(argv=None) -> int:
    opts = build_parser().parse_args(argv)
    return int(opts.func(opts))


if __name__ == "__main__":
    raise SystemExit(main())
