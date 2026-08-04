#!/usr/bin/env python3
"""
Convert a pair of BS-seq and oxBS-seq BigWig files (per-CpG methylation rate,
0..1, single-base intervals) into a 5mC BigWig by same-position subtraction:

        5mC_rate(chr, pos)  =  BS_rate(chr, pos)  -  oxBS_rate(chr, pos)

What is conserved
-----------------
- Only sites present in BOTH inputs are emitted. Sites missing in either file
  are dropped (5mC is undefined without the oxBS counterfactual). Use
  ``--strict`` to additionally drop negatives that come from noisy low-coverage
  positions (rare; default is to clip at 0 so the output stays a valid rate).

I/O
---
- Input  : two ``*.bw`` files. Both must be single-base intervals (start+1 == end).
- Output : ``--out`` 5mC BigWig (same chroms / sizes as input). An optional
  ``.bed.gz`` of the same positions+values is also written next to ``--out``
  unless ``--no-bedgraph`` is set.

Performance
-----------
- Streams both files chrom-by-chrom; iterates the 1-bp intervals with a
  two-pointer merge on ``start`` (the position set is sorted inside each bw).
  Memory is O(chrom).
- Tested on GSE214845 (mm10) BS/oxBS (~14M / ~8.7M intervals) — full run is a
  few minutes per sample.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time

import pyBigWig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def open_maybe_gz(path: str, mode: str = "wt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def chrom_sizes_from_bw(bw: pyBigWig.pyBigWig) -> list[tuple[str, int]]:
    return list(bw.chroms().items())


def write_bedgraph(
    chrom: str,
    intervals: list[tuple[int, float]],
    size: int,
    fh,
) -> None:
    """Write tabixed bedGraph lines: chrom  start  end  value."""
    for start, val in intervals:
        end = start + 1
        fh.write(f"{chrom}\t{start}\t{end}\t{val:.6f}\n")


# --------------------------------------------------------------------------- #
# core merge
# --------------------------------------------------------------------------- #

def merge_subtract(
    bs_path: str,
    ox_path: str,
    out_bw_path: str,
    bedgraph_path: str | None,
    strict: bool,
    clip_zero: bool,
) -> dict:
    bs = pyBigWig.open(bs_path)
    ox = pyBigWig.open(ox_path)
    if bs.isBigBed() or ox.isBigBed():
        raise SystemExit("inputs must be bigWig (bigBed not supported)")

    chrom_sizes = chrom_sizes_from_bw(bs)
    ox_chroms = ox.chroms()

    # emit chrom order/sizes from BS; warn for ox-only chroms
    missing_in_ox = [c for c, _ in chrom_sizes if c not in ox_chroms]
    if missing_in_ox:
        print(
            f"[warn] {len(missing_in_ox)} chrom(s) in BS but missing from oxBS "
            f"(no overlap possible); first: {missing_in_ox[:5]}",
            file=sys.stderr,
        )

    out = pyBigWig.open(out_bw_path, "w")
    out.addHeader(chrom_sizes)

    bg_fh = open_maybe_gz(bedgraph_path, "wt") if bedgraph_path else None

    n_kept = n_clipped_neg = n_skipped_neg = n_dropped_only_one = 0
    t0 = time.time()

    for chrom, size in chrom_sizes:
        if chrom not in ox_chroms:
            continue  # nothing to subtract against
        bs_it = bs.intervals(chrom, 0, size)
        ox_it = ox.intervals(chrom, 0, size)

        # materialise because generators can't be rewound
        bs_ivs = [(s, v) for s, _e, v in bs_it]
        ox_ivs = [(s, v) for s, _e, v in ox_it]

        out_ivs: list[tuple[int, int, float]] = []
        i = j = 0
        while i < len(bs_ivs) and j < len(ox_ivs):
            bs_s, bs_v = bs_ivs[i]
            ox_s, ox_v = ox_ivs[j]
            if bs_s == ox_s:
                val = bs_v - ox_v
                if val < 0:
                    if strict:
                        n_skipped_neg += 1
                    elif clip_zero:
                        val = 0.0  # clamp noisy negatives to 0; still kept
                        out_ivs.append((bs_s, bs_s + 1, val))
                        n_clipped_neg += 1
                    else:
                        n_skipped_neg += 1
                else:
                    out_ivs.append((bs_s, bs_s + 1, val))
                    n_kept += 1
                i += 1
                j += 1
            elif bs_s < ox_s:
                # BS-only site; oxBS needed to compute 5mC → drop
                n_dropped_only_one += 1
                i += 1
            else:
                n_dropped_only_one += 1
                j += 1

        # tails: any remaining intervals have no overlap
        # (we already accumulated "only-one" drops in the loop above; just drain)
        # n_dropped_only_one already counts only the ones we walked past with a
        # pending counterpart on the other side, but the trailing tail is also
        # "single-side". Track those too:
        # — simpler: we can count now.
        n_dropped_only_one += (len(bs_ivs) - i) + (len(ox_ivs) - j)

        if out_ivs:
            starts = [s for s, _e, _v in out_ivs]
            vals = [v for _s, _e, v in out_ivs]
            # pyBigWig wants (chrom, starts, values, span=1) for single-base entries
            out.addEntries(chrom, starts, values=vals, span=1)
            if bg_fh is not None:
                write_bedgraph(chrom, [(s, v) for s, _e, v in out_ivs],
                               size, bg_fh)

    out.close()
    bs.close()
    ox.close()
    if bg_fh is not None:
        bg_fh.close()

    return {
        "n_5mc_sites": n_kept + n_clipped_neg,
        "n_5mc_clipped_neg_to_zero": n_clipped_neg,
        "n_5mc_positive": n_kept,
        "n_dropped_single_side": n_dropped_only_one,
        "n_negative_dropped": n_skipped_neg,
        "elapsed_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="5mC = BS − oxBS, position-wise, on per-CpG BigWig inputs."
    )
    ap.add_argument("--bs", required=True, help="BS-seq bigWig (per-CpG rate)")
    ap.add_argument("--oxbs", required=True, help="oxBS-seq bigWig (per-CpG rate)")
    ap.add_argument("--out", required=True, help="output 5mC bigWig path")
    ap.add_argument("--bedgraph", default=None,
                    help="optional bedGraph path (.gz ok) of 5mC values")
    ap.add_argument("--no-bedgraph", action="store_true",
                    help="skip bedGraph emission even if --bedgraph is set")
    ap.add_argument("--strict", action="store_true",
                    help="drop sites where BS<oxBS (default: clip to 0)")
    args = ap.parse_args()

    bedgraph = None if args.no_bedgraph else args.bedgraph

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    stats = merge_subtract(
        bs_path=args.bs,
        ox_path=args.oxbs,
        out_bw_path=args.out,
        bedgraph_path=bedgraph,
        strict=args.strict,
        clip_zero=not args.strict,
    )
    print(f"[done] wrote {args.out}  →  {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
