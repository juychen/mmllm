"""Block masking: hide contiguous CpG runs (min 100 bp) until 15% of CpGs are covered.

Motivation
----------
The +/-50 bp protocol selects 15% of CpG *sites* but, because each hole is 101 bp
wide, it hides ~62% of the CpGs in a window.  This module implements the
alternative the user asked for:

    pick a CpG -> expand it into one contiguous masked block of at least
    ``min_block_bp`` -> keep doing that until the total number of masked CpGs
    reaches ``cpg_fraction`` (default 0.15) of the window.

So the *currency* is the number of CpGs hidden, not the number of holes.

API mirrors data.generate_cpg_pair_mask:
    hide, loss = generate_cpg_block_mask(base_ids, cpg_fraction=0.15,
                                         min_block_bp=100, max_block_bp=400, seed=None)

Run as a script to draw the schematic:
    python mask_blocks.py --out deliverables/block_mask_schematic.png
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from data import find_forward_cpg_positions


def generate_cpg_block_mask(
    base_ids_tensor: torch.Tensor,
    cpg_fraction: float = 0.15,
    min_block_bp: int = 100,
    max_block_bp: int = 400,
    seed: int | None = None,
    max_attempts: int = 500,
    length_mode: str = "adaptive",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous-block CpG mask.

    Args:
        base_ids_tensor: (N, L) long tensor, 0=A 1=C 2=G 3=T.
        cpg_fraction: fraction of the window's CpGs that end up masked.
        min_block_bp / max_block_bp: bounds of the block length.
        length_mode: 'adaptive' sizes each block to the CpGs still needed;
            'uniform' draws the length uniformly from [min_block_bp, max_block_bp].
        seed: RNG seed (shared draw per window, like the other mask functions).

    Returns:
        hide: (N, L, 1) float 1.0 over every base inside a masked block.
        loss: (N, L, 1) float 1.0 only at the forward C of each masked CpG.
    """
    n, length = base_ids_tensor.shape
    min_block_bp = max(1, int(min_block_bp))
    max_block_bp = max(min_block_bp, int(max_block_bp))
    cpg_c = find_forward_cpg_positions(base_ids_tensor)  # (N, L) bool

    hide = torch.zeros((n, length, 1), dtype=torch.float32)
    loss = torch.zeros((n, length, 1), dtype=torch.float32)
    rng = np.random.RandomState(seed)

    for i in range(n):
        idx = torch.nonzero(cpg_c[i], as_tuple=False).squeeze(-1).numpy()
        if idx.size == 0:
            continue
        target = max(1, int(round(idx.size * float(cpg_fraction))))
        covered = np.zeros(idx.size, dtype=bool)
        n_covered = 0
        attempts = 0
        while n_covered < target and attempts < max_attempts:
            attempts += 1
            j = rng.randint(idx.size)
            if covered[j]:
                continue
            p = int(idx[j])
            # Size the block so that it covers roughly the CpGs still needed.
            # Local spacing is estimated from the neighbours of p; the result is
            # clamped to [min_block_bp, max_block_bp] so the "at least 100 bp"
            # rule always holds.
            pos = np.sort(idx)
            k = int(np.searchsorted(pos, p))
            gaps = []
            if k > 0:
                gaps.append(p - int(pos[k - 1]))
            if k + 1 < pos.size:
                gaps.append(int(pos[k + 1]) - p)
            spacing = float(np.median(gaps)) if gaps else 20.0
            spacing = max(5.0, spacing)
            needed = max(1, target - n_covered)
            if length_mode == "uniform":
                blk = int(rng.randint(min_block_bp, max_block_bp + 1))
            else:
                blk = int(round(needed * spacing))
            blk = max(min_block_bp, min(max_block_bp, blk))
            left = int(rng.randint(0, max(1, blk - 1)))
            lo = max(0, p - left)
            hi = min(length, lo + blk)
            inside = (idx >= lo) & (idx < hi)
            newly = inside & (~covered)
            if not newly.any():
                continue
            covered |= inside
            n_covered = int(covered.sum())
            hide[i, lo:hi, 0] = 1.0
            loss[i, np.unique(idx[newly]), 0] = 1.0
    return hide, loss


def coverage_stats(base_ids: torch.Tensor, hide: torch.Tensor, loss: torch.Tensor) -> dict:
    """Return span% (bases hidden) and CpG% (CpGs hidden) for a batch."""
    cpg = find_forward_cpg_positions(base_ids)
    span = hide[..., 0].mean().item()
    total_cpg = cpg.sum().item()
    hidden_cpg = (cpg & (hide[..., 0] > 0)).sum().item()
    return {
        "span_pct": 100.0 * span,
        "cpg_pct": 100.0 * hidden_cpg / max(1, total_cpg),
        "n_cpg": int(total_cpg),
        "n_hidden": int(hidden_cpg),
    }


def _synthetic_window(length: int = 16000, seed: int = 0) -> torch.Tensor:
    """A synthetic cCRE-like 16 kb window (~140 CpGs, clustered like a CpG island).

    Real cCRE windows carry ~141 CpGs, most of them in dense clusters with
    ~18 bp spacing.  That density is what makes a +/-50 bp hole remove far more
    CPGs than the 15% of selected sites suggests, so the schematic has to
    reproduce it.
    """
    rng = np.random.RandomState(seed)
    bases = np.array([0, 2, 3], dtype=np.int64)  # A, G, T (no stray C)
    seq = rng.choice(bases, size=length).astype(np.int64)
    positions = []
    for k in range(8):  # 8 islands x 15 CpGs at ~18 bp spacing = 120
        p = 800 + k * 1900 + rng.randint(0, 30)
        for _ in range(15):
            positions.append(p)
            p += rng.randint(15, 22)
    for p in rng.choice(np.arange(200, length - 200, 90), size=20, replace=False):
        positions.append(int(p))  # sparse CpGs elsewhere = 140 total
    for p in positions:
        if 1 < p < length - 2:
            seq[p], seq[p + 1] = 1, 2
    return torch.from_numpy(seq).unsqueeze(0)


def block_stats(base_ids: torch.Tensor, cpg_fraction=0.15, min_block_bp=100,
                max_block_bp=400, length_mode="adaptive", seed=0):
    """Return (masked_cpg_pct, span_pct, mean_blocks, mean_cpg_per_block)."""
    hide, loss = generate_cpg_block_mask(base_ids, cpg_fraction, min_block_bp,
                                         max_block_bp, seed=seed, length_mode=length_mode)
    cpg = find_forward_cpg_positions(base_ids)
    n = base_ids.shape[0]
    cpg_pct = (cpg & (hide[..., 0] > 0)).sum(-1).float() / cpg.sum(-1).float()
    span_pct = hide[..., 0].mean(-1)
    blocks = []
    for i in range(n):
        h = hide[i, :, 0].numpy()
        runs = int(np.sum((h[1:] > 0) & (h[:-1] == 0)) + (1 if h[0] > 0 else 0))
        cov = int((cpg[i] & (hide[i, :, 0] > 0)).sum())
        blocks.append((runs, cov / max(1, runs)))
    runs = np.array([b[0] for b in blocks])
    per = np.array([b[1] for b in blocks])
    return dict(cpg_pct=float(cpg_pct.mean()), span_pct=float(span_pct.mean()),
                blocks=float(runs.mean()), cpg_per_block=float(per.mean()))


def draw_schematic(out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from data import generate_pretraining_cpg_mask, generate_cpg_pair_mask

    base = _synthetic_window()
    length = base.shape[1]
    cpg = find_forward_cpg_positions(base)[0].numpy().nonzero()[0]

    schemes = []
    single = generate_pretraining_cpg_mask(base, mask_fraction=0.15, seed=7)[..., 0]
    schemes.append(("A  single site   (15% of CpGs)", single, (single > 0)))
    pair = generate_cpg_pair_mask(base, mask_fraction=0.15, neighborhood_bp=50, seed=7)
    schemes.append(("B  +/-50 bp hole (15% of CpGs)", pair[0][..., 0], pair[1][..., 0] > 0))
    hide, loss = generate_cpg_block_mask(base, cpg_fraction=0.15, min_block_bp=100, max_block_bp=400, seed=7)
    schemes.append(("C  contiguous block, min 100 bp  (15% of CpGs)", hide[..., 0], loss[..., 0] > 0))

    navy, maroon, amber, grey = "#14213D", "#A50021", "#C26A00", "#9AA0A6"
    fig, axes = plt.subplots(3, 1, figsize=(13, 5.4), sharex=True)
    for ax, (title, hide_row, loss_row) in zip(axes, schemes):
        ax.set_xlim(0, length)
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(0, color=grey, lw=2.5, zorder=1, solid_capstyle="butt")
        # CpG ticks
        ax.vlines(cpg, -0.16, 0.16, color=navy, lw=1.4, zorder=3)
        # masked bands
        h = np.asarray(hide_row.detach().cpu()).reshape(-1)
        in_run = False
        for x in range(length + 1):
            on = x < length and h[x] > 0
            if on and not in_run:
                start, in_run = x, True
            elif not on and in_run:
                ax.add_patch(Rectangle((start, -0.55), x - start, 1.1, facecolor=amber,
                                       alpha=0.28, edgecolor=amber, lw=1.0, zorder=2))
                in_run = False
        # loss positions
        pos = np.nonzero(np.asarray(loss_row.detach().cpu()).reshape(-1))[0]
        ax.plot(pos, np.zeros_like(pos) + 0.0, "o", ms=4.0, color=maroon, zorder=4)
        st = coverage_stats(base, hide_row.unsqueeze(-1), loss_row.unsqueeze(-1).float())
        ax.set_title(title, loc="left", fontsize=10.5, color=navy, pad=4)
        ax.text(length * 0.995, 0.62,
                f"masked CpG {st['cpg_pct']:.0f}%   span {st['span_pct']:.1f}%",
                ha="right", va="center", fontsize=9.5, color=maroon)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])

    axes[-1].set_xlabel(f"position in a {length:,} bp window   (| = CpG,  shaded = hidden,  dot = loss position)",
                        fontsize=9.5, color=grey)
    fig.suptitle("Three masking schemes on the same window", x=0.012, ha="left",
                 fontsize=13, color=navy, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, facecolor="white")
    print(f"[saved] {out_path}")
    for title, hide_row, loss_row in schemes:
        st = coverage_stats(base, hide_row.unsqueeze(-1), loss_row.unsqueeze(-1).float())
        print(f"{title:52s} cpG={st['cpg_pct']:5.1f}%  span={st['span_pct']:5.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/junyichen/code/mmllm/deliverables/block_mask_schematic.png")
    args = ap.parse_args()
    draw_schematic(args.out)


def generate_cpg_mixed_budget_mask(
    base_ids_tensor: torch.Tensor,
    cpg_fraction: float = 0.15,
    widths: tuple = (0, 10, 25, 50, 100),
    seed: int | None = None,
    slack: float = 0.10,
    max_attempts: int = 400,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed hole widths under a fixed budget of HIDDEN CpGs.

    Unlike ``generate_cpg_pair_mask_mixed`` (which fixes the number of *scored*
    CpGs and lets the holes swallow whatever they overlap), this version spends
    a budget: at most ``cpg_fraction`` of the window's CpGs may end up hidden.
    Each hole still draws its width at random from ``widths``; a width that would
    blow the budget is retried a few times, then the largest width that still
    fits is used.

    Returns:
        hide: (N, L, 1) 1.0 over every base inside a hole.
        loss: (N, L, 1) 1.0 at the forward C of every hidden CpG.
    """
    n, length = base_ids_tensor.shape
    cpg_c = find_forward_cpg_positions(base_ids_tensor)
    hide = torch.zeros((n, length, 1), dtype=torch.float32)
    loss = torch.zeros((n, length, 1), dtype=torch.float32)
    rng = np.random.RandomState(seed)
    widths = tuple(int(w) for w in widths)

    for i in range(n):
        idx = torch.nonzero(cpg_c[i], as_tuple=False).squeeze(-1).numpy()
        if idx.size == 0:
            continue
        target = max(1, int(round(idx.size * float(cpg_fraction))))
        budget = int(round(target * (1.0 + slack)))
        covered = np.zeros(idx.size, dtype=bool)
        attempts = 0
        while covered.sum() < target and attempts < max_attempts:
            attempts += 1
            j = rng.randint(idx.size)
            if covered[j]:
                continue
            p = int(idx[j])
            order = list(widths)
            rng.shuffle(order)
            chosen = None
            for w in order:
                lo, hi = max(0, p - w), min(length, p + w + 1)
                ins = (idx >= lo) & (idx < hi)
                if (covered | ins).sum() <= budget:
                    chosen = w
                    break
            if chosen is None:
                continue
            lo, hi = max(0, p - chosen), min(length, p + chosen + 1)
            ins = (idx >= lo) & (idx < hi)
            covered |= ins
            hide[i, lo:hi, 0] = 1.0
            loss[i, idx[ins], 0] = 1.0
    return hide, loss
