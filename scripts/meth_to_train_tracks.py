#!/usr/bin/env python3
"""
Generate training-ready 5mC and 5hmC bedGraph tracks from the GSE214845
``.meth.txt.gz`` files (BS + oxBS paired per sample).

Output for each sample, in ``--out-dir``:

    <SAMPLE>_5mc.bedGraph   chrom  start  end  pct(0..100)        bgzipped + .tbi
    <SAMPLE>_5hmc.bedGraph  chrom  start  end  pct(0..100)        bgzipped + .tbi

Where:
  * 5mC = oxBS methylation rate   (oxBS measures only 5mC, no subtraction)
  * 5hmC = BS rate − oxBS rate    (BS = 5mC+5hmC, oxBS = 5mC)

Both are written in **percentage 0..100** to match ``data.py``'s convention
(it divides track values by 100 before forming the multi-task target).
Negative 5hmC values are kept as-is — the model can learn "low 5hmC"
rather than clipping to 0 which would bias the prior.

Each output is bgzipped and tabix-indexed so it can be consumed directly
by ``data.py`` via ``pysam.TabixFile``.

Coverage filter: ``--min-cov`` (default 5) on BOTH BS and oxBS for the
5hmC track; for 5mC only the oxBS side is filtered.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Iterator

import numpy as np
import pyBigWig


# --------------------------------------------------------------------------- #
# streaming reader
# --------------------------------------------------------------------------- #

def stream_chroms(
    path: str,
) -> Iterator[tuple[str, list[tuple[int, float, int]]]]:
    """Yield ``(chrom, [(pos, rate, cov), ...])`` chrom-by-chrom."""
    cur: str | None = None
    buf: list[tuple[int, float, int]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            chrom, pos, _strand, _ctx, rate, cov = line.rstrip("\n").split("\t")
            pi, rf, ci = int(pos), float(rate), int(cov)
            if cur is None:
                cur = chrom
            if chrom != cur:
                yield cur, buf
                cur = chrom
                buf = []
            buf.append((pi, rf, ci))
    if buf:
        yield cur, buf


# --------------------------------------------------------------------------- #
# core merge — writes two plain bedGraph files (5mc and 5hmc)
# --------------------------------------------------------------------------- #

def write_train_tracks(
    bs_path: str,
    ox_path: str,
    sample_name: str,
    out_dir: str,
    header_from_bw: str | None,
    min_cov: int,
    keep_negative: bool,
) -> dict:
    if header_from_bw:
        ref = pyBigWig.open(header_from_bw)
        chrom_sizes = list(ref.chroms().items())
        ref.close()
    else:
        # derive from BS .meth (no header bw given)
        sizes: dict[str, int] = {}
        with gzip.open(bs_path, "rt") as f:
            for line in f:
                chrom, pos, *_ = line.rstrip("\n").split("\t")
                end = int(pos) + 1
                if end > sizes.get(chrom, 0):
                    sizes[chrom] = end
        chrom_sizes = sorted(sizes.items())

    # Sort chrom_sizes ascending to match the order in the meth files
    chrom_sizes_sorted = sorted(chrom_sizes, key=lambda x: x[0])

    five_mc_path   = os.path.join(out_dir, f"{sample_name}_5mc.bedGraph")
    five_hmc_path  = os.path.join(out_dir, f"{sample_name}_5hmc.bedGraph")

    n_5mc = n_5hmc = n_5hmc_neg = 0
    t0 = time.time()

    bs_iter = stream_chroms(bs_path)
    ox_iter = stream_chroms(ox_path)
    cur_ox_chrom, cur_ox_list = next(ox_iter)

    five_mc_fh  = open(five_mc_path, "wt")
    five_hmc_fh = open(five_hmc_path, "wt")

    for bs_chrom, bs_list in bs_iter:
        # advance ox so cur_ox_chrom == bs_chrom
        while cur_ox_chrom is not None and cur_ox_chrom < bs_chrom:
            try:
                cur_ox_chrom, cur_ox_list = next(ox_iter)
            except StopIteration:
                cur_ox_chrom = None
                cur_ox_list = []
        if cur_ox_chrom != bs_chrom:
            continue

        ox_list = cur_ox_list
        # Two-pointer merge.
        i = j = 0
        while i < len(bs_list) and j < len(ox_list):
            bp, br, bc = bs_list[i]
            op, orr, oc = ox_list[j]

            # 5mC track: every oxBS-pass site qualifies (5mC = oxBS rate)
            if op == bp and oc >= min_cov:
                five_mc_fh.write(
                    f"{bs_chrom}\t{bp}\t{bp+1}\t{orr*100:.2f}\n"
                )
                n_5mc += 1
            elif bp < op:
                pass  # bs only, no oxBS → no 5mC measurement

            # 5hmC track: requires both BS and oxBS at same pos with cov>=min_cov
            if bp == op and bc >= min_cov and oc >= min_cov:
                hmc = (br - orr) * 100.0
                if (not keep_negative) and hmc < 0:
                    # Optional: clamp negatives. Off by default — let the
                    # model learn "low 5hmC" instead of fake-zero.
                    hmc = 0.0
                    n_5hmc_neg += 1
                elif hmc < 0:
                    n_5hmc_neg += 1
                five_hmc_fh.write(
                    f"{bs_chrom}\t{bp}\t{bp+1}\t{hmc:.2f}\n"
                )
                n_5hmc += 1

            if bp == op:
                i += 1
                j += 1
            elif bp < op:
                i += 1
            else:
                j += 1

        cur_ox_list = []  # consumed for this chrom

    five_mc_fh.close()
    five_hmc_fh.close()

    # bgzip + tabix for each output
    def bgzip_and_index(plain: str) -> None:
        bg = plain + ".gz"
        # bgzip writes to <plain>.gz
        subprocess.run(["bgzip", "-f", plain], check=True)
        # tabix index — bedGraph format preset: 1-based start not needed;
        # bedGraph is 0-based; use -p bed
        subprocess.run(
            ["tabix", "-f", "-p", "bed", bg],
            check=True,
        )

    bgzip_and_index(five_mc_path)
    bgzip_and_index(five_hmc_path)

    return {
        "sample": sample_name,
        "n_5mc_sites": n_5mc,
        "n_5hmc_sites": n_5hmc,
        "n_5hmc_negative_or_clipped": n_5hmc_neg,
        "min_cov": min_cov,
        "out_files": [
            five_mc_path + ".gz",
            five_hmc_path + ".gz",
            five_mc_path + ".gz.tbi",
            five_hmc_path + ".gz.tbi",
        ],
        "elapsed_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--bs", required=True,
                    help="BS .meth.txt.gz (GSE214845 supplementary)")
    ap.add_argument("--oxbs", required=True,
                    help="oxBS .meth.txt.gz")
    ap.add_argument("--sample", required=True,
                    help="sample name (used in output filenames)")
    ap.add_argument("--out-dir", required=True,
                    help="directory for the bedGraph outputs")
    ap.add_argument("--header-bw", default=None,
                    help="bigwig to copy chrom header from (recommended)")
    ap.add_argument("--min-cov", type=int, default=5,
                    help="minimum reads on BOTH BS and oxBS for 5hmC "
                         "(5mC only filters oxBS). Default 5.")
    ap.add_argument("--clip-negative", action="store_true",
                    help="clip negative 5hmC values to 0 (off by default)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stats = write_train_tracks(
        bs_path=args.bs,
        ox_path=args.oxbs,
        sample_name=args.sample,
        out_dir=args.out_dir,
        header_from_bw=args.header_bw,
        min_cov=args.min_cov,
        keep_negative=not args.clip_negative,
    )
    print(f"[done] {args.sample}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())