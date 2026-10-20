#!/usr/bin/env python
"""Build a SHARED holdout + leakage-free train pools for a fair A/B comparison.

Motivation
----------
To compare "trained on cCRE regions" vs "trained on the whole genome" fairly you
need (a) the SAME test set for both, and (b) no model training on anything near
it.  This script produces exactly that:

  test.bed                      the shared TEST set  = the reference windows that
                                fall inside the holdout (identical for every model)
  <name>_trainpool.bed          windows of that set with the holdout AND a
                                `--buffer` around it removed  -> use for train/val
  holdout_regions.bed           the raw holdout regions
  holdout_plus_buffer.bed       holdout expanded by the buffer (the exclusion zone)

Windows are expanded to `--target-length` exactly like `data.py` does
(``center ± L/2`` when shorter than L), so the overlap test reflects the 16 kb
window the model actually reads.

Typical use
-----------
  python make_holdout_bed.py \
    --holdout-chroms chr9 chr19 \
    --windows ccre=/data2st1/junyi/generegion_vM23/cCRE_cpg.bed \
    --windows wg=output/whole_genome_16kb_nonoverlap_allchr.bed \
    --test-windows ccre \
    --target-length 16384 --buffer 16384 \
    --out-dir output/holdout_chr9_chr19

then train BOTH models with a train pool that excludes the holdout and evaluate
BOTH on the same test.bed (via `--test-bed <out>/test.bed`).

Notes
-----
* For whole-chromosome holdouts there is no cross-chromosome adjacency, so the
  buffer mainly matters for block holdouts; it is kept configurable regardless.
* The buffer windows are dropped from training AND are not test -> pure guard.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_windows(path: str) -> pd.DataFrame:
    """chr / start / end (1-based inclusive), matching data.py's BED reader."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2],
                     names=["chr", "start", "end"])
    df["chr"] = df["chr"].astype(str)
    df["start"] = df["start"].astype(int) + 1     # BED 0-based -> 1-based inclusive
    df["end"] = df["end"].astype(int)
    return df


def expand_windows(df: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """Reproduce data.py: short windows expand to `target_length` bp around center."""
    df = df.copy()
    length = df["end"] - df["start"] + 1
    center = (df["start"] + df["end"]) // 2
    half = target_length // 2
    short = length < target_length
    df["start_exp"] = np.where(short, center - half, df["start"])
    df["end_exp"] = np.where(short, center + half - 1, df["end"])
    return df


def merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def index_regions(regions: dict[str, list[tuple[int, int]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    idx = {}
    for c, ivs in regions.items():
        merged = merge_intervals(ivs)
        if merged:
            idx[c] = (np.array([s for s, _ in merged], dtype=np.int64),
                      np.array([e for _, e in merged], dtype=np.int64))
    return idx


def overlap_mask(chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                 idx: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """True where [start,end] overlaps any (disjoint, sorted) interval of idx."""
    mask = np.zeros(len(starts), dtype=bool)
    for c in pd.unique(chroms):
        m = chroms == c
        if c not in idx:
            continue
        iv_s, iv_e = idx[c]
        # merged intervals are disjoint+sorted -> the only candidate is the last
        # interval whose start <= window.end
        pos = np.searchsorted(iv_s, ends[m], side="right") - 1
        hit = pos >= 0
        res = np.zeros(pos.shape, dtype=bool)
        if hit.any():
            res[hit] = iv_e[pos[hit]] >= starts[m][hit]
        mask[m] = res
    return mask


def write_bed(path: Path, chroms, starts, ends) -> None:
    with open(path, "w") as fh:
        for c, s, e in zip(chroms, starts, ends):
            fh.write(f"{c}\t{int(s)-1}\t{int(e)}\n")   # back to BED 0-based half-open


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", action="append", required=True, metavar="NAME=PATH",
                   help="Window set(s) to filter. Repeatable. The NAME is used in output filenames.")
    p.add_argument("--test-windows", default=None,
                   help="Which NAME defines the shared TEST set (its windows inside the holdout). "
                        "Default: the first --windows.")
    p.add_argument("--holdout-chroms", nargs="+", default=None,
                   help="Whole chromosomes to hold out, e.g. chr9 chr19.")
    p.add_argument("--holdout-bed", default=None,
                   help="Alternative: a BED of holdout blocks (chr/start/end).")
    p.add_argument("--target-length", type=int, default=16384)
    p.add_argument("--buffer", type=int, default=16384,
                   help="Extra bp to exclude around the holdout (guard zone). Default 16384 (1 window).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--extra-test", action="append", default=None, metavar="NAME",
                   help="Also write test_<NAME>.bed for these window sets (2nd evaluation view), "
                        "e.g. --extra-test wg for a genome-wide test view.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sets: dict[str, pd.DataFrame] = {}
    for spec in args.windows:
        if "=" not in spec:
            raise SystemExit(f"--windows needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        sets[name] = read_windows(path)
        print(f"[load] {name:10s} {len(sets[name]):>7,} windows  <- {path}")

    # ---- holdout regions -----------------------------------------------------
    if args.holdout_bed:
        hb = read_windows(args.holdout_bed)
        regions: dict[str, list[tuple[int, int]]] = {}
        for c, s, e in zip(hb["chr"], hb["start"], hb["end"]):
            regions.setdefault(c, []).append((int(s), int(e)))
        print(f"[holdout] from BED: {len(hb):,} regions, {len(regions)} chromosomes")
    elif args.holdout_chroms:
        chroms = set(args.holdout_chroms)
        seen = set()
        for df in sets.values():
            seen |= set(df["chr"].unique())
        missing = chroms - seen
        if missing:
            raise SystemExit(f"holdout chromosomes not present in any window set: {sorted(missing)}")
        regions = {}
        for c in chroms:
            hi = max(int(df.loc[df["chr"] == c, "end"].max()) for df in sets.values())
            regions[c] = [(1, hi)]
        print(f"[holdout] whole chromosomes: {sorted(chroms)}")
    else:
        raise SystemExit("need --holdout-chroms or --holdout-bed")

    n_bp = sum(e - s + 1 for ivs in regions.values() for s, e in ivs)
    print(f"[holdout] total span {n_bp/1e6:.1f} Mb across {len(regions)} chromosomes")

    raw_idx = index_regions(regions)
    buf_regions = {c: [(s - args.buffer, e + args.buffer) for s, e in ivs] for c, ivs in regions.items()}
    buf_idx = index_regions(buf_regions)

    # reference (test-defining) set
    ref_name = args.test_windows or next(iter(sets))
    if ref_name not in sets:
        raise SystemExit(f"--test-windows {ref_name!r} is not one of --windows {list(sets)}")

    print("\n" + "=" * 78)
    print(f"{'set':10s} {'total':>8} {'buffered+holdout':>18} {'trainpool':>10} {'in-holdout':>11}")
    print("-" * 78)

    results = {}
    for name, df in sets.items():
        ex = expand_windows(df, args.target_length)
        chroms = ex["chr"].to_numpy()
        s, e = ex["start_exp"].to_numpy(), ex["end_exp"].to_numpy()
        excl = overlap_mask(chroms, s, e, buf_idx)          # drop from training
        inho = overlap_mask(chroms, s, e, raw_idx)          # inside the raw holdout
        results[name] = dict(df=df, chroms=chroms, s=s, e=e, excl=excl, inho=inho)

        trainpool = ~excl
        write_bed(out / f"{name}_trainpool.bed", chroms[trainpool], s[trainpool], e[trainpool])
        print(f"{name:10s} {len(df):>8,} {int(excl.sum()):>18,} {int(trainpool.sum()):>10,} {int(inho.sum()):>11,}")

    # ---- the shared TEST set = reference windows inside the holdout ----------
    ref = results[ref_name]
    test_mask = ref["inho"]
    write_bed(out / "test.bed", ref["chroms"][test_mask], ref["s"][test_mask], ref["e"][test_mask])

    # ---- the guard zone, for the record -------------------------------------
    write_bed(out / "holdout_regions.bed",
              [c for c, ivs in regions.items() for _ in ivs],
              [s for ivs in regions.values() for s, _ in ivs],
              [e for ivs in regions.values() for _, e in ivs])
    write_bed(out / "holdout_plus_buffer.bed",
              [c for c, ivs in buf_regions.items() for _ in ivs],
              [s for ivs in buf_regions.values() for s, _ in ivs],
              [e for ivs in buf_regions.values() for _, e in ivs])

    print("=" * 78)
    print(f"[test] shared TEST set = '{ref_name}' windows in holdout: {int(test_mask.sum()):,}")
    print(f"       -> {out/'test.bed'}")

    for extra in (args.extra_test or []):
        if extra not in results:
            raise SystemExit(f"--extra-test {extra!r} is not one of --windows {list(sets)}")
        er = results[extra]
        em = er["inho"]
        write_bed(out / f"test_{extra}.bed", er["chroms"][em], er["s"][em], er["e"][em])
        print(f"[test] 2nd view '{extra}' windows in holdout: {int(em.sum()):,}  -> {out/f'test_{extra}.bed'}")

    # ---- sanity: test must never overlap any train pool ----------------------
    print("\n[check] test windows overlapping each model's train pool (must be 0):")
    ok = True
    t_chroms, t_s, t_e = ref["chroms"][test_mask], ref["s"][test_mask], ref["e"][test_mask]
    for name, r in results.items():
        tp = ~r["excl"]
        pool: dict[str, list[tuple[int, int]]] = {}
        for cc, ss, ee in zip(r["chroms"][tp], r["s"][tp], r["e"][tp]):
            pool.setdefault(cc, []).append((int(ss), int(ee)))
        ov = overlap_mask(t_chroms, t_s, t_e, index_regions(pool))
        print(f"  {name:10s} overlap = {int(ov.sum())}")
        ok &= int(ov.sum()) == 0

    print("\n" + ("[OK] holdout is clean for every model." if ok else "[!!] LEAKAGE detected."))
    print(f"[done] outputs in {out}")
    print(f"       use:  --test-bed {out/'test.bed'}")
    print(f"       and train each model on  <out>/<name>_trainpool.bed")


if __name__ == "__main__":
    main()
