#!/usr/bin/env python3
"""Log1p -> cap at p99 -> scale to [0,1] for ATAC / RNA bigWigs.

For every (region_condition, track):
    x' = log1p(max(x, 0))
    p99 = 99th percentile of x' over positive bins (default)
    y  = min(x' / p99, 1)          # --scale-to-one (default)
    y  = min(x', p99)              # --no-scale-to-one

The p99 is estimated from a histogram of log1p values over the whole track, so
it is a global statistic (not per 16-kb window).  Output goes to
<out-dir>/ATAC_<group>_scaled.bw and RNA_<group>_scaled.bw.

Important: these outputs are already 0-1.  If you feed them to mmllm, set
--atac-scaling none --rna-scaling none, otherwise the dataset will apply a
second per-window min-max and undo the global scaling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import pyBigWig

GROUPS = ["AMY_MC", "AMY_MW", "HIP_MC", "HIP_MW", "PFC_MC", "PFC_MW"]
CHUNK = 5_000_000
HIST_BINS = 20_000
HIST_MAX = 10.0  # log1p(22026); CPM tracks are far below this


def iter_intervals(bw: pyBigWig.pyBigWig, chrom: str, chunk: int = CHUNK):
    length = bw.chroms()[chrom]
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        iv = bw.intervals(chrom, start, end)
        if iv:
            yield iv


def estimate_p99(path: str, percentile: float, include_zeros: bool, chroms=None):
    """Histogram-based global percentile of log1p values; returns (p99, n_bins)."""
    bw = pyBigWig.open(path)
    chroms_dict = bw.chroms()
    keys = [c for c in chroms_dict if (chroms is None or c in chroms)]
    edges = np.linspace(0.0, HIST_MAX, HIST_BINS + 1)
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    n = 0
    for c in keys:
        for iv in iter_intervals(bw, c):
            vals = np.asarray([x[2] for x in iv], dtype=np.float64)
            if not include_zeros:
                vals = vals[vals > 0.0]
            if vals.size == 0:
                continue
            x = np.clip(np.log1p(np.maximum(vals, 0.0)), 0.0, HIST_MAX)
            h, _ = np.histogram(x, bins=edges)
            hist += h
            n += int(vals.size)
    bw.close()
    if n == 0 or hist.sum() == 0:
        return 0.0, 0
    cum = np.cumsum(hist)
    target = float(percentile) / 100.0 * float(cum[-1])
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), HIST_BINS - 1)
    return float(edges[idx]), n


def write_scaled(in_path: str, out_path: str, p99: float, scale_to_one: bool, chroms=None):
    bw = pyBigWig.open(in_path)
    chroms_dict = bw.chroms()
    out = pyBigWig.open(out_path, "w")
    out.addHeader(list(chroms_dict.items()), maxZooms=0)
    keys = [c for c in chroms_dict if (chroms is None or c in chroms)]
    n_written = 0
    for c in keys:
        for iv in iter_intervals(bw, c):
            starts = np.asarray([x[0] for x in iv], dtype=np.int32)
            ends = np.asarray([x[1] for x in iv], dtype=np.int32)
            vals = np.asarray([x[2] for x in iv], dtype=np.float64)
            vals = np.log1p(np.maximum(vals, 0.0))
            if p99 > 0.0:
                if scale_to_one:
                    vals = np.minimum(vals / p99, 1.0)
                else:
                    vals = np.minimum(vals, p99)
            else:
                vals = np.zeros_like(vals)
            vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0)
            out.addEntries([c] * len(iv), starts, ends=ends, values=vals.astype(float))
            n_written += len(iv)
    out.close()
    bw.close()
    return n_written


def main():
    ap = argparse.ArgumentParser(description="log1p + p99 cap + scale ATAC/RNA bigWigs")
    ap.add_argument("--out-dir", default="/data1st1/junyi/methdata/scaled_tracks")
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--tracks", nargs="+", default=["atac", "rna"], choices=["atac", "rna"])
    ap.add_argument("--atac-root", default="/data1st1/junyi/methdata/atac")
    ap.add_argument("--rna-root", default="/data2st1/junyi/output/sn0615")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--scale-to-one", action=argparse.BooleanOptionalAction, default=True,
                    help="divide by p99 to get 0-1 (default); use --no-scale-to-one to keep log1p")
    ap.add_argument("--include-zeros", action="store_true",
                    help="include zero bins in the percentile (default: positive bins only)")
    ap.add_argument("--chroms", nargs="+", default=None,
                    help="optional chromosome subset (for smoke tests)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only estimate p99, do not write")
    ap.add_argument("--tag", default="", help="worker tag for the manifest filename")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    manifest = {
        "timestamp": ts,
        "out_dir": args.out_dir,
        "percentile": args.percentile,
        "scale_to_one": args.scale_to_one,
        "include_zeros": args.include_zeros,
        "chroms": args.chroms,
        "tag": args.tag,
        "tracks": {},
    }

    for group in args.groups:
        region = group.split("_")[0]
        for track in args.tracks:
            if track == "atac":
                in_path = os.path.join(args.atac_root, f"{group}_track.bw")
                out_name = f"ATAC_{group}_scaled.bw"
            else:
                in_path = os.path.join(args.rna_root, f"BULK_{region}", f"{group}.bw")
                out_name = f"RNA_{group}_scaled.bw"
            out_path = os.path.join(args.out_dir, out_name)

            print(f"\n[{track}] {in_path}")
            if not os.path.exists(in_path):
                print(f"  [skip] missing: {in_path}")
                continue
            if args.skip_existing and os.path.exists(out_path):
                print(f"  [skip] exists: {out_path}")
                continue

            t0 = time.time()
            p99, n = estimate_p99(in_path, args.percentile, args.include_zeros, args.chroms)
            print(f"  p99(log1p)={p99:.6f}  p99(raw)={math.expm1(p99):.4f}  n_bins={n}")
            entry = {
                "input": in_path,
                "output": out_path,
                "p99_log1p": p99,
                "p99_raw": math.expm1(p99),
                "n_bins": n,
                "seconds": None,
            }
            if not args.dry_run:
                n_written = write_scaled(in_path, out_path, p99, args.scale_to_one, args.chroms)
                entry["n_written"] = n_written
                entry["seconds"] = round(time.time() - t0, 1)
                print(f"  wrote {n_written} intervals -> {out_path}  ({entry['seconds']}s)")
            manifest["tracks"][f"{track}:{group}"] = entry

    suffix = f"_{args.tag}" if args.tag else f"_{os.getpid()}"
    manifest_path = os.path.join(args.out_dir, f"preprocess_manifest_{ts}{suffix}.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\n[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
