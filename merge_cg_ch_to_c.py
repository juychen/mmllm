#!/usr/bin/env python
"""
Merge per-context (CG, CH) bedGraph files into a single whole-cytosine (.C) track.

Input  (for one condition × region):
    /path/.../{condition}_{region}.CG.m.bedGraph[.gz]
    /path/.../{condition}_{region}.CH.m.bedGraph[.gz]
    /path/.../{condition}_{region}.CG.h.bedGraph[.gz]
    /path/.../{condition}_{region}.CH.h.bedGraph[.gz]

Output:
    /path/.../{condition}_{region}.C.m.bedGraph[.gz]   (merged 5mC)
    /path/.../{condition}_{region}.C.h.bedGraph[.gz]   (merged 5hmC)

Strategy: position-wise UNION of CG and CH.
- Each unique (chr, start, end) interval is kept exactly once.
- The 4th column (value) is the source's rate.
- CG and CH occupy disjoint sites (CpG vs non-CpG), so values from each
  context are kept separately and concatenated; both end up in one file.

This is the safest merge: it preserves the original modification rate at
each position without averaging across contexts (which would be
meaningless because CG and CH rates differ in magnitude by ~10-50x).
"""

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd


def read_bedgraph(path: Path) -> pd.DataFrame:
    """Read a bedGraph file (possibly gzipped) into a DataFrame."""
    open_fn = gzip.open if str(path).endswith(".gz") else open
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "value"],
        dtype={"chr": str, "start": int, "end": int, "value": float},
    )
    df["chr"] = df["chr"].astype(str)
    return df


def write_bedgraph(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to a (gzipped) bedGraph file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        df.to_csv(path, sep="\t", index=False, header=False,
                  compression="gzip", float_format="%.6f")
    else:
        df.to_csv(path, sep="\t", index=False, header=False, float_format="%.6f")


def merge_pair(cg_path: Path, ch_path: Path, output_path: Path, label: str) -> int:
    """Merge one pair of (CG, CH) bedGraph files into a single C track.

    Returns the number of intervals written.
    """
    cg_df = read_bedgraph(cg_path) if cg_path.exists() else pd.DataFrame(columns=["chr", "start", "end", "value"])
    ch_df = read_bedgraph(ch_path) if ch_path.exists() else pd.DataFrame(columns=["chr", "start", "end", "value"])

    if cg_df.empty and ch_df.empty:
        print(f"  [{label}] skipped: neither CG nor CH file found")
        return 0

    # Union of positions, keeping the original value
    merged = pd.concat([cg_df, ch_df], ignore_index=True)
    # Drop duplicates (in case the same site appears in both — shouldn't happen
    # biologically since CG and CH are mutually exclusive, but be defensive)
    merged = merged.drop_duplicates(subset=["chr", "start", "end"], keep="first")
    # Sort by genomic position for downstream tools
    merged = merged.sort_values(["chr", "start", "end"]).reset_index(drop=True)

    write_bedgraph(merged, output_path)
    print(f"  [{label}] {len(merged):,} intervals  →  {output_path}")
    return len(merged)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-context (CG, CH) bedGraph files into a single .C track.",
    )
    parser.add_argument(
        "--input-dir",
        default="/data2st1/junyi/output/llm0401/processed_meth",
        help="Directory containing {condition}_{region}.{CG|CH}.{m|h}.bedGraph[.gz] files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as --input-dir)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["MC", "MW"],
        help="Conditions to process (default: MC MW)",
    )
    parser.add_argument(
        "--regions", nargs="+", default=["AMY", "HIP", "PFC"],
        help="Brain regions to process (default: AMY HIP PFC)",
    )
    parser.add_argument(
        "--contexts", nargs="+", default=["CG", "CH"],
        help="Input contexts to merge (default: CG CH)",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=["m", "h"],
        help="Modalities: m=5mC, h=5hmC (default: m h)",
    )
    parser.add_argument(
        "--no-gzip", action="store_true",
        help="Write plain .bedGraph instead of .bedGraph.gz",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip if output file already exists",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gz_suffix = "" if args.no_gzip else ".gz"

    total = 0
    for condition in args.conditions:
        for region in args.regions:
            print(f"\n=== {condition} × {region} ===")
            for modality in args.modalities:
                label = f"{condition}_{region}.C.{modality}"
                cg_path = input_dir / f"{condition}_{region}.CG.{modality}.bedGraph"
                ch_path = input_dir / f"{condition}_{region}.CH.{modality}.bedGraph"
                # Try .gz variants if plain files don't exist
                if not cg_path.exists():
                    cg_gz = cg_path.with_suffix(".bedGraph.gz")
                    if cg_gz.exists():
                        cg_path = cg_gz
                if not ch_path.exists():
                    ch_gz = ch_path.with_suffix(".bedGraph.gz")
                    if ch_gz.exists():
                        ch_path = ch_gz

                output_path = output_dir / f"{condition}_{region}.C.{modality}.bedGraph{gz_suffix}"

                if args.skip_existing and output_path.exists():
                    print(f"  [{label}] skipped: {output_path} already exists")
                    continue

                n = merge_pair(cg_path, ch_path, output_path, label)
                total += n

    print(f"\nDone. Wrote {total:,} total intervals across {len(args.conditions) * len(args.regions) * len(args.modalities)} output files.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
