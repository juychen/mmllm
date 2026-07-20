"""Generate ablation comparison bar plot for AMY_MC cross_hyena runs.

Reads the per-ablation `_summary.tsv` files produced by `run_atac_ablation.py`
across the two ATAC ablation sweeps, merges them, and writes:

  output/summary/ablation_barplot_R2.png
  output/summary/ablation_barplot_pearson.png
  output/summary/ablation_barplot_combined.png
  output/summary/ablation_summary.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output" / "summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_ROOTS = [
    REPO_ROOT / "output" / "atac_ablation" / "full_mirror_modelb",
    REPO_ROOT / "output" / "atac_ablation" / "seq_only_all_three",
]

# Standalone JSON results for the phastCons and sn baselines (logs in
# /home/junyichen/logs/run_all_m5c_{phascon,sn}.sh.*.out, but the canonical
# metrics live in the per-run results.json files written by the training
# pipeline). These share the m5C query / ATAC-like context configuration
# (so they're directly comparable to m5c_atac).
EXTRA_RESULTS: dict[str, Path] = {
    "phascon": REPO_ROOT
    / "output"
    / "AMY_MC"
    / "cCRE_cpg"
    / "phascon"
    / "2026-07-14-16-13-22_m5c_query_sequence_phascon_modelb_cross_hyena_results.json",
    "sn": REPO_ROOT
    / "output"
    / "AMY_MC"
    / "cCRE_cpg"
    / "sn"
    / "2026-07-16-13-21-29_m5c_query_sequence_phascon_modelb_cross_hyena_results.json",
}

# Display order / colors (fixed categorical order, no cycling).
ABLATION_ORDER = [
    "m5c_atac",      # baseline: m5C query, ATAC context
    "atac_m5c",      # swapped roles
    "m5c_only",      # m5C only
    "atac_only",     # ATAC only
    "seq_query",     # sequence query with both context tracks
    "all_three",     # m5C+ATAC as one query track
    "seq_only",      # sequence only (no epigenomic input)
    "phascon",       # m5C query, phastCons context track (AMY_MC)
    "sn",            # m5C query, sn0601 context track (AMY_MC)
]

ABLATION_COLORS = {
    "m5c_atac":   "#1b9e77",
    "atac_m5c":   "#7570b3",
    "m5c_only":   "#66a61e",
    "atac_only":  "#d95f02",
    "seq_query":  "#e6ab02",
    "all_three":  "#e7298a",
    "seq_only":   "#666666",
    "phascon":    "#386cb0",
    "sn":         "#fdc086",
}

QUERY_CONTEXT_LABELS = {
    "m5c_atac":   ("m5C",       "ATAC"),
    "atac_m5c":   ("ATAC",      "m5C"),
    "m5c_only":   ("m5C",       "—"),
    "atac_only":  ("ATAC",      "—"),
    "seq_query":  ("seq",       "ATAC+m5C"),
    "all_three":  ("m5C+ATAC",  "—"),
    "seq_only":   ("seq",       "—"),
    "phascon":    ("m5C",       "phastCons"),
    "sn":         ("m5C",       "sn0601"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _normalize_context(value: object) -> str:
    """Normalize context column to display string ('' when empty)."""
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", ""} else s


def load_ablation_summaries() -> pd.DataFrame:
    """Load and merge per-sweep `_summary.tsv` files into a single frame."""
    frames: list[pd.DataFrame] = []
    for root in ABLATION_ROOTS:
        path = root / "_summary.tsv"
        if not path.exists():
            print(f"[skip] missing summary: {path}")
            continue
        df = pd.read_csv(path, sep="\t")
        df = df.rename(columns={"best_val_pearson": "best_val_pearsonr"})
        # Keep only completed rows; failed/blank runs have no numeric metrics.
        mask = df["ablation"].astype(str).str.contains("FAILED", na=False)
        df = df.loc[~mask].copy()
        df["context"] = df["context"].map(_normalize_context)
        df["sweep"] = root.name
        # Some sweeps report pearson as pearsonr; already renamed above.
        frames.append(df)

    # Standalone JSON results for phastCons / sn baselines.
    for name, json_path in EXTRA_RESULTS.items():
        if not json_path.exists():
            print(f"[skip] missing extra json: {json_path}")
            continue
        payload = json.loads(json_path.read_text())
        result = (payload.get("results") or [{}])[0]
        q_label = "phastCons" if name == "phascon" else "sn0601"
        frames.append(
            pd.DataFrame(
                [
                    {
                        "ablation": name,
                        "query": "m5c",
                        "context": q_label,
                        "best_val_loss": result.get("best_val_loss"),
                        "best_val_r2": result.get("best_val_r2"),
                        "best_val_pearsonr": result.get("best_val_pearsonr"),
                        "sweep": name,
                    }
                ]
            )
        )

    if not frames:
        raise SystemExit("No ablation summaries found.")

    merged = pd.concat(frames, ignore_index=True)
    merged["ablation"] = merged["ablation"].astype(str).str.strip()
    # Prefer most-recent sweep on duplicates (seq_only_all_three supersedes).
    merged = merged.drop_duplicates(subset="ablation", keep="last")
    merged = merged.set_index("ablation").reindex(ABLATION_ORDER).dropna(how="all")
    return merged.reset_index()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _bar(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    labels = df["ablation"].tolist()
    values = df[metric].astype(float).to_numpy()
    colors = [ABLATION_COLORS.get(name, "#999999") for name in labels]
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    # Direct labels on each bar with the numeric value.
    ymax = float(np.nanmax(values)) if len(values) else 1.0
    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.005 * ymax,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222",
        )

    ax.set_xticks(x)
    tick_labels = []
    for name in labels:
        q, c = QUERY_CONTEXT_LABELS.get(name, ("", ""))
        tick_labels.append(f"{name}\nq: {q}\nc: {c}")
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.tick_params(axis="x", which="both", pad=4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(ymax * 1.18, 0.05))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _sorted(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return df sorted by `metric` descending, breaking ties by ablation name."""
    return df.sort_values(
        by=[metric, "ablation"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def render_combined(df: pd.DataFrame) -> Path:
    sorted_by_r2 = _sorted(df, "best_val_r2")
    sorted_by_pearson = _sorted(df, "best_val_pearsonr")
    n_bars = len(df)
    width = max(11.5, 1.55 * n_bars + 4.5)
    fig, axes = plt.subplots(1, 2, figsize=(width, 5.6))
    _bar(axes[0], sorted_by_r2, "best_val_r2", "Validation R²")
    _bar(axes[1], sorted_by_pearson, "best_val_pearsonr", "Validation Pearson r")
    fig.suptitle("AMY_MC ablation comparison (sorted desc, cross_hyena, single seed)", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = OUTPUT_DIR / "ablation_barplot_combined_sorted.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_single(df: pd.DataFrame, metric: str, ylabel: str) -> Path:
    sorted_df = _sorted(df, metric)
    n_bars = len(df)
    width = max(7.5, 1.4 * n_bars + 3.0)
    fig, ax = plt.subplots(figsize=(width, 5.0))
    _bar(ax, sorted_df, metric, ylabel)
    ax.set_title(f"Ablation {ylabel} sorted desc (cross_hyena, single seed)")
    fig.tight_layout()
    out = OUTPUT_DIR / f"ablation_barplot_{metric.replace('best_val_', '')}_sorted.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_ablation_summaries()
    csv_out = OUTPUT_DIR / "ablation_summary.csv"
    df_out = df[
        ["ablation", "query", "context", "sweep", "best_val_loss", "best_val_r2", "best_val_pearsonr"]
    ].copy()
    df_out.to_csv(csv_out, index=False)

    png_r2 = render_single(df, "best_val_r2", "R²")
    png_p = render_single(df, "best_val_pearsonr", "Pearson r")
    png_combined = render_combined(df)

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {png_r2}")
    print(f"Wrote: {png_p}")
    print(f"Wrote: {png_combined}")


if __name__ == "__main__":
    main()