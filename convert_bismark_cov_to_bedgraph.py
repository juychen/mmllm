#!/usr/bin/env python
"""
Convert paired Bismark coverage files (WGBS + TabSeq) to 5mC and 5hmC bedGraph tracks.

Biological rationale:
  - WGBS (bisulfite-seq): 5mC + 5hmC are both protected from conversion
  - TabSeq (TET-assisted bisulfite-seq): TET oxidizes 5mC→5caC (converted),
    5hmC is glucosylated and protected
  - Therefore: WGBS = 5mC + 5hmC,  TabSeq = 5hmC
  - True 5mC = WGBS - TabSeq (clamped to >= 0)

Input:
  GSM4154671_7w_cortex_rep1_wgbs.bismark.cov.gz    (WGBS: 5mC + 5hmC)
  GSM4154672_7w_cortex_rep1_tabSeq.bismark.cov.gz   (TabSeq: 5hmC)

Bismark .cov.gz format (6 columns, no header):
  chr  start  end  methylation_pct  count_methylated  count_unmethylated

Output (bedGraph, 4 columns):
  5mC bedGraph: chr start end (wgbs_pct - tabseq_pct, clamped >= 0)
  5hmC bedGraph: chr start end tabseq_pct

Usage:
  python convert_bismark_cov_to_bedgraph.py \
      --wgbs GSM4154671_7w_cortex_rep1_wgbs.bismark.cov.gz \
      --tabseq GSM4154672_7w_cortex_rep1_tabSeq.bismark.cov.gz \
      --output-dir ./processed_meth \
      --sample-name 7w_cortex_rep1
"""

import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COV_COLUMNS = ["chr", "start", "end", "methylation_pct", "count_meth", "count_unmeth"]


def read_bismark_cov(path: str) -> pd.DataFrame:
    """Read a Bismark .cov.gz file into a DataFrame."""
    print(f"Reading {path} ...")
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COV_COLUMNS,
        dtype={
            "chr": str,
            "start": int,
            "end": int,
            "methylation_pct": float,
            "count_meth": int,
            "count_unmeth": int,
        },
    )
    df["chr"] = df["chr"].astype(str)
    print(f"  {len(df):,} sites loaded")
    return df


def compute_5mc_5hmc(
    wgbs_df: pd.DataFrame,
    tabseq_df: pd.DataFrame,
    min_coverage: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute 5mC and 5hmC bedGraph DataFrames from WGBS and TabSeq coverage.

    Strategy: merge on genomic position (chr, start, end).
    - 5hmC = TabSeq methylation_pct  (directly measured)
    - 5mC  = WGBS methylation_pct - TabSeq methylation_pct  (clamped >= 0)

    Sites present in only one file are kept as-is (treated as 0 in the other).
    """
    wgbs = wgbs_df.set_index(["chr", "start", "end"])
    tabseq = tabseq_df.set_index(["chr", "start", "end"])

    # Optional coverage filter
    if min_coverage > 0:
        wgbs = wgbs[wgbs["count_meth"] + wgbs["count_unmeth"] >= min_coverage]
        tabseq = tabseq[tabseq["count_meth"] + tabseq["count_unmeth"] >= min_coverage]

    # Merge on genomic position; fill missing with 0
    merged = pd.DataFrame(index=wgbs.index.union(tabseq.index))
    merged["wgbs_pct"] = wgbs["methylation_pct"].reindex(merged.index).fillna(0.0)
    merged["tabseq_pct"] = tabseq["methylation_pct"].reindex(merged.index).fillna(0.0)

    # 5hmC = TabSeq
    hm5c = merged[["tabseq_pct"]].copy()
    hm5c.columns = ["value"]
    hm5c = hm5c[hm5c["value"] > 0].reset_index()  # drop zero sites to save space
    hm5c = hm5c.sort_values(["chr", "start", "end"])

    # 5mC = WGBS - TabSeq (clamped >= 0)
    merged["m5c_value"] = (merged["wgbs_pct"] - merged["tabseq_pct"]).clip(lower=0.0)
    m5c = merged[["m5c_value"]].copy()
    m5c.columns = ["value"]
    m5c = m5c[m5c["value"] > 0].reset_index()
    m5c = m5c.sort_values(["chr", "start", "end"])

    print(f"  Merged positions: {len(merged):,}")
    print(f"  WGBS-only sites: {(merged['tabseq_pct'] == 0).sum():,}")
    print(f"  TabSeq-only sites: {(merged['wgbs_pct'] == 0).sum():,}")
    print(f"  Shared sites: {((merged['wgbs_pct'] > 0) & (merged['tabseq_pct'] > 0)).sum():,}")
    print(f"  5mC sites (after subtraction): {len(m5c):,}")
    print(f"  5hmC sites: {len(hm5c):,}")

    return m5c, hm5c


def write_bedgraph(df: pd.DataFrame, path: Path, header: bool = False) -> None:
    """Write DataFrame to bedGraph file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, header=header, float_format="%.6f")
    print(f"  Wrote {len(df):,} intervals → {path}")


def bgzip_and_tabix(bg_path: Path) -> None:
    """Compress bedGraph with bgzip and create tabix index."""
    gz_path = Path(str(bg_path) + ".gz")
    print(f"  Compressing: {bg_path} → {gz_path}")
    subprocess.run(
        ["bgzip", "-f", str(bg_path)],
        check=True,
    )
    print(f"  Indexing: {gz_path}")
    subprocess.run(
        ["tabix", "-p", "bed", str(gz_path)],
        check=True,
    )
    print(f"  Done: {gz_path} + {gz_path}.tbi")


def main():
    parser = argparse.ArgumentParser(
        description="Convert paired WGBS + TabSeq Bismark cov files to 5mC/5hmC bedGraph tracks.",
    )
    parser.add_argument(
        "--wgbs", required=True,
        help="Path to WGBS Bismark .cov.gz file (5mC + 5hmC)",
    )
    parser.add_argument(
        "--tabseq", required=True,
        help="Path to TabSeq Bismark .cov.gz file (5hmC only)",
    )
    parser.add_argument(
        "--output-dir", default="./processed_meth",
        help="Output directory for bedGraph files (default: ./processed_meth)",
    )
    parser.add_argument(
        "--sample-name", default="sample",
        help="Sample name prefix for output files (default: sample)",
    )
    parser.add_argument(
        "--min-coverage", type=int, default=0,
        help="Minimum read coverage per site (default: 0 = no filter)",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Skip bgzip + tabix indexing (keep plain .bedGraph)",
    )
    args = parser.parse_args()

    for fpath in [args.wgbs, args.tabseq]:
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Read
    wgbs_df = read_bismark_cov(args.wgbs)
    tabseq_df = read_bismark_cov(args.tabseq)

    # Step 2: Compute 5mC and 5hmC
    print("\n--- Computing 5mC and 5hmC ---")
    m5c_df, hm5c_df = compute_5mc_5hmc(wgbs_df, tabseq_df, min_coverage=args.min_coverage)

    # Step 3: Write bedGraph
    print("\n--- Writing output ---")
    m5c_path = output_dir / f"{args.sample_name}.5mC.bedGraph"
    hm5c_path = output_dir / f"{args.sample_name}.5hmC.bedGraph"

    write_bedgraph(m5c_df, m5c_path, header=["#chr", "start", "end", "value"])
    write_bedgraph(hm5c_df, hm5c_path, header=["#chr", "start", "end", "value"])

    # Step 4: Compress and index (optional)
    if not args.no_compress:
        print("\n--- Compressing and indexing ---")
        bgzip_and_tabix(m5c_path)
        bgzip_and_tabix(hm5c_path)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output directory: {output_dir.resolve()}")
    if not args.no_compress:
        print("Ready-to-use files for mmllm pipeline:")
        print(f"  --m5c-bedgraph  {m5c_path}.gz")
        print(f"  --hm5c-bedgraph  {hm5c_path}.gz")
    print("=" * 60)


if __name__ == "__main__":
    main()
