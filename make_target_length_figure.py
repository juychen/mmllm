#!/usr/bin/env python
"""Plot the initial target_length sweep results for the 5mC→5hmC paper.

Data sources (all on /data1st2/zhangyr):
  test_results/target_length_{1024,2048,4096,8192,16384,32768}/AMY_MC/*_5mc_to_5hmc_results.json

Also draws the cross-region numbers at 1024/2048/4096/8192 for HIP/PFC × MC/MW
where available, so the figure also tells the cross-region story.

Outputs to /data2st1/junyi/output/mmllm/output/summary/:
  - target_length_sweep_R2.png  (R² vs target_length, AMY_MC 5mc→5hmc)
  - target_length_sweep_pearson.png  (Pearson r vs target_length)
  - target_length_sweep_combined.png  (2-panel R² + r)
  - target_length_sweep_cross_region_R2.png  (heatmap regions × length)
  - target_length_sweep_summary.csv  (clean long-form table)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/data1st2/zhangyr/data/mmllm/test_results")
OUT_DIR = Path("/data2st1/junyi/output/mmllm/output/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]
REGIONS = ["AMY_MC", "AMY_MW", "HIP_MC", "HIP_MW", "PFC_MC", "PFC_MW"]
TASK = "5mc_to_5hmc"
PALETTE = {
    "AMY_MC": "#1f77b4",
    "AMY_MW": "#ff7f0e",
    "HIP_MC": "#2ca02c",
    "HIP_MW": "#d62728",
    "PFC_MC": "#9467bd",
    "PFC_MW": "#8c564b",
}


def load_json(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["results"]


def load_region_length(region: str, length: int) -> list[dict]:
    """Return list of result dicts for (region, length) if available."""
    candidates = list((DATA_ROOT / f"target_length_{length}" / region).glob(f"*_{TASK}_results.json"))
    if not candidates:
        return []
    return load_json(candidates[0])


def main() -> None:
    # --- AMY_MC main sweep ---
    rows = []
    for L in LENGTHS:
        results = load_region_length("AMY_MC", L)
        for r in results:
            rows.append({
                "region": "AMY_MC",
                "target_length": L,
                "n_dmrs": r["num_dmrs"],
                "best_epoch": r["best_epoch"],
                "val_loss": r["best_val_loss"],
                "val_r2": r["best_val_r2"],
                "val_pearsonr": r["best_val_pearsonr"],
            })
    df_amy = pd.DataFrame(rows)
    df_amy.to_csv(OUT_DIR / "target_length_sweep_AMY_MC_full.csv", index=False)

    # --- Cross-region at 4096 (most-complete panel) ---
    cross_rows = []
    for region in REGIONS:
        for L in LENGTHS:
            results = load_region_length(region, L)
            if not results:
                continue
            # take the largest n_dmrs run for each (region, length)
            best = max(results, key=lambda x: x["num_dmrs"])
            cross_rows.append({
                "region": region,
                "target_length": L,
                "n_dmrs": best["num_dmrs"],
                "val_loss": best["best_val_loss"],
                "val_r2": best["best_val_r2"],
                "val_pearsonr": best["best_val_pearsonr"],
            })
    df_cross = pd.DataFrame(cross_rows)
    df_cross.to_csv(OUT_DIR / "target_length_sweep_cross_region.csv", index=False)
    print(f"[saved] {OUT_DIR / 'target_length_sweep_cross_region.csv'} ({len(df_cross)} rows)")

    # --- Figure 1: R² vs target_length (AMY_MC, sample_size fan-out) ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    sample_sizes = sorted(df_amy["n_dmrs"].unique())
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(sample_sizes):
        sub = df_amy[df_amy["n_dmrs"] == n].sort_values("target_length")
        ax.plot(sub["target_length"], sub["val_r2"], "o-",
                color=cmap(i / max(1, len(sample_sizes) - 1)),
                label=f"n={n:,}", linewidth=1.6, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS)
    ax.set_xticklabels([str(L) for L in LENGTHS])
    ax.set_xlabel("target_length (bp)")
    ax.set_ylabel("Validation R²")
    ax.set_title("AMY_MC  ·  5mC + sequence  →  5hmC\n(target_length sweep, baseline model, May 2026)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2, frameon=False)
    # mark the sweet spot
    sweet_spot = 16384
    ax.axvline(sweet_spot, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.annotate("sweet spot\n16384 bp",
                xy=(sweet_spot, df_amy.query("target_length == 16384 and n_dmrs == 78891")["val_r2"].iloc[0]),
                xytext=(sweet_spot * 0.45, 0.56),
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.6),
                fontsize=9, color="red")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_length_sweep_R2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'target_length_sweep_R2.png'}")

    # --- Figure 2: Pearson r vs target_length ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    for i, n in enumerate(sample_sizes):
        sub = df_amy[df_amy["n_dmrs"] == n].sort_values("target_length")
        ax.plot(sub["target_length"], sub["val_pearsonr"], "o-",
                color=cmap(i / max(1, len(sample_sizes) - 1)),
                label=f"n={n:,}", linewidth=1.6, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS)
    ax.set_xticklabels([str(L) for L in LENGTHS])
    ax.set_xlabel("target_length (bp)")
    ax.set_ylabel("Validation Pearson r")
    ax.set_title("AMY_MC  ·  5mC + sequence  →  5hmC\n(target_length sweep, baseline model)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_length_sweep_pearson.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'target_length_sweep_pearson.png'}")

    # --- Figure 3: 2-panel combined (R² + r) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130, sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "val_r2", "Validation R²"),
        (axes[1], "val_pearsonr", "Validation Pearson r"),
    ]:
        for i, n in enumerate(sample_sizes):
            sub = df_amy[df_amy["n_dmrs"] == n].sort_values("target_length")
            ax.plot(sub["target_length"], sub[metric], "o-",
                    color=cmap(i / max(1, len(sample_sizes) - 1)),
                    label=f"n={n:,}", linewidth=1.8, markersize=7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS)
        ax.set_xticklabels([str(L) for L in LENGTHS])
        ax.set_xlabel("target_length (bp)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.axvline(16384, color="red", linestyle="--", alpha=0.4)
    fig.suptitle("Initial target_length sweep  ·  AMY_MC  ·  5mC + sequence → 5hmC\n"
                 "(baseline model, mask=cpg_forward, RC aug, hidden=64, seed=7)",
                 fontsize=12, y=1.02)
    axes[0].legend(loc="lower right", fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_length_sweep_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'target_length_sweep_combined.png'}")

    # --- Figure 4: cross-region heatmap (R²) ---
    pivot = df_cross.pivot_table(index="region", columns="target_length", values="val_r2", aggfunc="max")
    pivot = pivot.reindex(REGIONS)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0.40, vmax=0.60)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("target_length (bp)")
    ax.set_ylabel("Region")
    ax.set_title("Cross-region R²  ·  5mC + sequence → 5hmC\n"
                 "(baseline model, max n_dmrs in each cell)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                txt, color = "—", "white"
            else:
                txt, color = f"{v:.3f}", ("white" if v < 0.52 else "black")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Validation R²")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_length_sweep_cross_region_R2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'target_length_sweep_cross_region_R2.png'}")

    # --- Summary CSV ---
    summary = df_amy[df_amy["n_dmrs"] == 78891].copy()
    summary["delta_r2_vs_1024"] = summary["val_r2"] - summary["val_r2"].iloc[0]
    summary["delta_r2_vs_prev"] = summary["val_r2"].diff()
    summary.to_csv(OUT_DIR / "target_length_sweep_summary.csv", index=False)
    print(f"[saved] {OUT_DIR / 'target_length_sweep_summary.csv'}")

    # --- Print summary table ---
    print("\n=== AMY_MC 5mc→5hmc, 78891 DMRs, by target_length ===")
    print(f"{'L':>6} {'epoch':>6} {'loss':>8} {'R²':>8} {'r':>8} {'ΔR² vs prev':>12}")
    prev = None
    for _, row in summary.iterrows():
        delta = "" if prev is None else f"{row['val_r2'] - prev:+.4f}"
        print(f"{int(row['target_length']):>6} {int(row['best_epoch']):>6} "
              f"{row['val_loss']:>8.2f} {row['val_r2']:>8.4f} {row['val_pearsonr']:>8.4f} {delta:>12}")
        prev = row["val_r2"]


if __name__ == "__main__":
    main()