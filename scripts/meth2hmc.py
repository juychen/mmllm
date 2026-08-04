#!/usr/bin/env python3
"""
Build a 5hmC BigWig from a pair of methPipe ``.meth.txt.gz`` files (BS + oxBS).

This is the count-aware cousin of ``bw_bs_oxbs_to_5mc.py``. The meth.txt.gz
files carry a coverage column (the bw alone doesn't), so we can:

  * apply a *coverage threshold* on both BS and oxBS (mirrors GSE214845's
    processing; default 5x, matching the published pipeline)
  * quantify per-CpG uncertainty with a Beta-Binomial posterior
    (5hmC's 95% HDI; sites whose HDI crosses 0 are written as NaN so the
    downstream analyst / peak caller can decide what to do with them — much
    safer than faking a 5hmC = 0)

Per-CpG ``.meth.txt.gz`` line is (deduplicated, sorted):

    chrom  pos  strand  context  rate  cov

The line is already strand-merged (one row per CpG, ``strand == "+"``).

Inputs required
---------------
* ``--bs``, ``--oxbs``           .meth.txt.gz files (same assembly)
* ``--header-bw`` (optional)     Any bigwig from this dataset — its chrom
                                 header is reused for the 5hmC bw.

Outputs
-------
* ``--out``                      5hmC BigWig (single-base; val in [0,1] or NaN)
* ``--out-bedgraph`` (optional)  bedGraph with extra cols ``5hmC bs_cov ox_cov``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from typing import Iterator

import numpy as np
import pyBigWig


# --------------------------------------------------------------------------- #
# streaming readers
# --------------------------------------------------------------------------- #

def stream_chroms(
    path: str,
) -> Iterator[tuple[str, list[tuple[int, float, int]]]]:
    """
    Yield ``(chrom, [(pos, rate, cov), ...])`` one chrom at a time, fully
    streaming. Each chrom is materialised before yield, so memory scales
    with the largest single chrom (~50 MB peak on chr1).
    """
    cur_chrom: str | None = None
    cur_list: list[tuple[int, float, int]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            chrom, pos, _strand, _ctx, rate, cov = line.rstrip("\n").split("\t")
            pos_i, rate_f, cov_i = int(pos), float(rate), int(cov)
            if cur_chrom is None:
                cur_chrom = chrom
            if chrom != cur_chrom:
                yield cur_chrom, cur_list
                cur_chrom = chrom
                cur_list = []
            cur_list.append((pos_i, rate_f, cov_i))
    if cur_list:
        yield cur_chrom, cur_list


def scan_max_pos(path: str) -> dict[str, int]:
    """One-shot scan: max 1-based end per chrom (sizes for the bw header)."""
    sizes: dict[str, int] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            chrom, pos, _strand, _ctx, _rate, _cov = line.rstrip("\n").split("\t")
            end = int(pos) + 1
            if end > sizes.get(chrom, 0):
                sizes[chrom] = end
    return sizes


# --------------------------------------------------------------------------- #
# per-CpG math
# --------------------------------------------------------------------------- #

def rate_to_counts(rate: float, cov: int) -> tuple[int, int]:
    n_m = int(round(rate * cov))
    n_m = max(0, min(n_m, cov))
    return n_m, cov - n_m


def diff_post_stats(
    n_m_bs: int, n_u_bs: int, n_m_ox: int, n_u_ox: int,
    n_samp: int = 2000, seed: int = 0,
) -> tuple[float, float, float]:
    """Posterior median + 95% HDI of (5hmC) via Beta-Binomial Monte-Carlo."""
    rng = np.random.default_rng(seed)
    bs = rng.beta(1 + n_m_bs, 1 + n_u_bs, size=n_samp)
    ox = rng.beta(1 + n_m_ox, 1 + n_u_ox, size=n_samp)
    diff = bs - ox
    return (
        float(np.median(diff)),
        float(np.quantile(diff, 0.025)),
        float(np.quantile(diff, 0.975)),
    )


# --------------------------------------------------------------------------- #
# core merge: two-pointer intersection on (pos)
# --------------------------------------------------------------------------- #

def merge_two_pointers(
    bs_list: list[tuple[int, float, int]],
    ox_list: list[tuple[int, float, int]],
    min_cov: int,
    use_bb: bool,
) -> Iterator[tuple[int, float, int, int]]:
    """
    Yield ``(pos, five_hmc_value, bs_cov, ox_cov)`` for CpGs present in both
    sides with ``cov >= min_cov``.

    Value semantics:
      * ``use_bb``: posterior median, OR ``NaN`` if the 95% HDI spans 0.
      * else:     raw ``bs_rate - ox_rate`` (kept as-is, may be negative).
    """
    i = j = 0
    while i < len(bs_list) and j < len(ox_list):
        bp, br, bc = bs_list[i]
        op, orr, oc = ox_list[j]
        if bp == op:
            if bc >= min_cov and oc >= min_cov:
                if use_bb:
                    nmb, nub = rate_to_counts(br, bc)
                    nmo, nuo = rate_to_counts(orr, oc)
                    med, lo, hi = diff_post_stats(nmb, nub, nmo, nuo)
                    val: float = float("nan") if lo <= 0 <= hi else med
                else:
                    val = br - orr
                yield bp, val, bc, oc
            i += 1
            j += 1
        elif bp < op:
            i += 1
        else:
            j += 1


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #

def build_5hmc_bw(
    bs_path: str,
    ox_path: str,
    out_bw_path: str,
    out_bedgraph_path: str | None,
    header_from_bw: str | None,
    min_cov: int,
    use_bb: bool,
) -> dict:
    # Chrom sizes: reuse an existing bw if provided (consistent across the
    # dataset); else derive by scanning BS once.
    if header_from_bw:
        ref = pyBigWig.open(header_from_bw)
        chrom_sizes = list(ref.chroms().items())
        ref.close()
    else:
        sizes = scan_max_pos(bs_path)
        chrom_sizes = sorted(sizes.items())
        # give a little headroom so entries aren't clipped at the chrom end
        chrom_sizes = [(c, sz) for c, sz in chrom_sizes]

    out = pyBigWig.open(out_bw_path, "w")
    out.addHeader(chrom_sizes)

    bg_fh = (
        gzip.open(out_bedgraph_path, "wt")
        if out_bedgraph_path and out_bedgraph_path.endswith(".gz")
        else (open(out_bedgraph_path, "wt") if out_bedgraph_path else None)
    )

    n_kept = n_nan_hdi = n_skipped_lowcov = 0
    t0 = time.time()

    bs_iter = stream_chroms(bs_path)
    ox_iter = stream_chroms(ox_path)
    # prime ox_iter
    cur_ox_chrom, cur_ox_list = next(ox_iter)

    for bs_chrom, bs_list in bs_iter:
        # advance ox so cur_ox_chrom == bs_chrom (or exhaust)
        while cur_ox_chrom is not None and cur_ox_chrom < bs_chrom:
            try:
                cur_ox_chrom, cur_ox_list = next(ox_iter)
            except StopIteration:
                cur_ox_chrom = None
                cur_ox_list = []
        if cur_ox_chrom != bs_chrom:
            # no oxBS for this BS chrom (e.g. JH584296.1)
            continue

        starts: list[int] = []
        vals: list[float] = []

        for pos, val, bs_c, ox_c in merge_two_pointers(
            bs_list, cur_ox_list, min_cov, use_bb,
        ):
            if val != val:  # NaN check
                n_nan_hdi += 1
                continue
            starts.append(pos)
            vals.append(val)
            n_kept += 1
            if bg_fh is not None:
                bg_fh.write(
                    f"{bs_chrom}\t{pos}\t{pos+1}\t{val:.6f}\t"
                    f"{bs_c}\t{ox_c}\n"
                )

        if starts:
            out.addEntries(bs_chrom, starts, values=vals, span=1)

        # consumed cur_ox_list; clear so next iter triggers re-advance
        cur_ox_list = []

    if bg_fh is not None:
        bg_fh.close()
    out.close()

    return {
        "n_5hmc_kept": n_kept,
        "n_5hmc_nan_hdi_crosses_0": n_nan_hdi,
        "min_cov": min_cov,
        "use_bb_posterior": use_bb,
        "elapsed_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--bs", required=True,
                    help="BS .meth.txt.gz with coverage column")
    ap.add_argument("--oxbs", required=True,
                    help="oxBS .meth.txt.gz with coverage column")
    ap.add_argument("--out", required=True, help="output 5hmC bigWig")
    ap.add_argument("--out-bedgraph", default=None,
                    help="optional bedGraph with extra cols (5hmC, bs_cov, ox_cov)")
    ap.add_argument("--header-bw", default=None,
                    help="any bigwig from same dataset to copy chrom header from "
                         "(recommended; else we derive from --bs)")
    ap.add_argument("--min-cov", type=int, default=5,
                    help="min reads on BOTH BS and oxBS (default 5, matches "
                         "GSE214845 published pipeline)")
    ap.add_argument("--bb-posterior", action="store_true",
                    help="use Beta-Binomial posterior; mark HDI-crosses-0 as NaN")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    stats = build_5hmc_bw(
        bs_path=args.bs,
        ox_path=args.oxbs,
        out_bw_path=args.out,
        out_bedgraph_path=args.out_bedgraph,
        header_from_bw=args.header_bw,
        min_cov=args.min_cov,
        use_bb=args.bb_posterior,
    )
    print(f"[done] {args.out}\n  {json.dumps(stats, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
