#!/usr/bin/env python3
"""Prepare GSE166423 (mouse Purkinje neurons) as mmllm fine-tune input.

Creates a NEW self-contained folder; raw GSE166423 files are untouched.

What it does
------------
1. 5mC/5hmC bigWigs: methylation values are 0-1 -> write scaled bigWigs x100
   (0-100, matching the cCRE pretraining scale).
2. ATAC bigWigs: copy as-is.  The training dataset applies per-window min-max
   (--atac-scaling minmax, the default), so raw coverage is correct.
3. RNA: GSE166423 has no RNA -> copy the mm10 zero bigWig for use with
   --drop-p-rna 1.0.
4. Regions: copy cCRE_cpg.bed and the cCRE test BED for a self-contained input.
5. Write manifest.csv / manifest.json.

Output
------
/data1st1/junyi/methdata/GSE166423_finetune_input/
  Purkinje_Adult_5mC_scaled100.bw
  Purkinje_Adult_5hmC_scaled100.bw
  Purkinje_P0_5mC_scaled100.bw ...
  Purkinje_Adult_ATAC.bw ...
  zero_RNA_mm10.bw
  regions/cCRE_cpg.bed
  regions/cpg_test.bed
  manifest.csv / manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime

import pyBigWig

BASE = "/data1st1/junyi/methdata/GSE166423_extracted/oxbs_bs_bigwigs"
ATAC_DIR = "/data1st1/junyi/methdata/GSE166423/bigwig"
OUT = "/data1st1/junyi/methdata/GSE166423_finetune_input"
ZERO_RNA = "/data1st1/junyi/methdata/GSE244251/zero_track_mm10.bw"
CCRE = "/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
TESTBED = "/home/junyichen/code/mmllm/output/holdout_testchroms/cpg_test.bed"

STAGES = {
    "Adult": ("purkinje_adult_methyl_CpG_merged.bw", "purkinje_adult_hydroxy_CpG_merged.bw", "Purkinje_Adult_atac.bw"),
    "P0": ("purkinje_p0_methyl_CpG_merged.bw", "purkinje_p0_hydroxy_CpG_merged.bw", "Purkinje_P0_atac.bw"),
    "P7": ("purkinje_p7_methyl_CpG_merged.bw", "purkinje_p7_hydroxy_CpG_merged.bw", "Purkinje_P7_atac.bw"),
}


def scale_bigwig(src: str, dst: str, scale: float = 100.0):
    """Write a bigWig with values multiplied by `scale`; preserve intervals."""
    bw = pyBigWig.open(src)
    chroms = bw.chroms()
    out = pyBigWig.open(dst, "w")
    out.addHeader(list(chroms.items()), maxZooms=0)
    n_total = 0
    for chrom in chroms:
        iv = bw.intervals(chrom)
        if not iv:
            continue
        starts = [x[0] for x in iv]
        ends = [x[1] for x in iv]
        vals = [float(x[2]) * scale for x in iv]
        out.addEntries([chrom] * len(iv), starts, ends=ends, values=vals)
        n_total += len(iv)
    out.close()
    bw.close()
    return n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    regions_dir = os.path.join(out, "regions")
    os.makedirs(regions_dir, exist_ok=True)

    manifest = {
        "dataset": "GSE166423",
        "description": "Mouse Purkinje neurons, 5mC/5hmC/ATAC, Adult/P0/P7",
        "created": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "scaling": {"5mC": "raw 0-1 -> x100 (0-100)", "5hmC": "raw 0-1 -> x100 (0-100)",
                     "ATAC": "raw coverage, use --atac-scaling minmax", "RNA": "zero, use --drop-p-rna 1.0"},
        "stages": {},
    }
    rows = []

    for stage, (m_src, h_src, a_src) in STAGES.items():
        m_src = os.path.join(BASE, m_src)
        h_src = os.path.join(BASE, h_src)
        a_src = os.path.join(ATAC_DIR, a_src)
        m_dst = os.path.join(out, f"Purkinje_{stage}_5mC_scaled100.bw")
        h_dst = os.path.join(out, f"Purkinje_{stage}_5hmC_scaled100.bw")
        a_dst = os.path.join(out, f"Purkinje_{stage}_ATAC.bw")

        for src in (m_src, h_src, a_src):
            if not os.path.exists(src):
                raise SystemExit(f"missing: {src}")

        if args.force or not os.path.exists(m_dst):
            print(f"[scale] {stage} 5mC -> {m_dst}")
            n = scale_bigwig(m_src, m_dst)
        else:
            print(f"[skip] {m_dst}")
            n = -1
        if args.force or not os.path.exists(h_dst):
            print(f"[scale] {stage} 5hmC -> {h_dst}")
            scale_bigwig(h_src, h_dst)
        else:
            print(f"[skip] {h_dst}")

        if args.force or not os.path.exists(a_dst):
            shutil.copy2(a_src, a_dst)
            print(f"[copy] {stage} ATAC -> {a_dst}")
        else:
            print(f"[skip] {a_dst}")

        manifest["stages"][stage] = {"5mC": m_dst, "5hmC": h_dst, "ATAC": a_dst}
        rows.append({"stage": stage, "5mC": m_dst, "5hmC": h_dst, "ATAC": a_dst,
                     "RNA": os.path.join(out, "zero_RNA_mm10.bw")})

    zero_dst = os.path.join(out, "zero_RNA_mm10.bw")
    if args.force or not os.path.exists(zero_dst):
        shutil.copy2(ZERO_RNA, zero_dst)
        print(f"[copy] zero RNA -> {zero_dst}")
    else:
        print(f"[skip] {zero_dst}")

    for src, name in ((CCRE, "cCRE_cpg.bed"), (TESTBED, "cpg_test.bed")):
        dst = os.path.join(regions_dir, name)
        if args.force or not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"[copy] regions -> {dst}")
        else:
            print(f"[skip] {dst}")

    manifest["zero_RNA"] = zero_dst
    manifest["regions"] = {"cCRE_cpg": os.path.join(regions_dir, "cCRE_cpg.bed"),
                            "cpg_test": os.path.join(regions_dir, "cpg_test.bed")}

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "5mC", "5hmC", "ATAC", "RNA"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
