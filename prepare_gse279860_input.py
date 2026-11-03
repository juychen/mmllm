#!/usr/bin/env python3
"""Convert GSE279860 (mouse cerebellum, Nanopore) BedMethyl to model input.

GSE279860 is aligned to **mm39**; this script:
  1. reads the 4 `*.masked.bed.gz` samples
  2. keeps CpG forward-C calls (`strand == '+'`, C followed by G in mm39)
  3. writes 5mC and 5hmC bedGraphs in mm39
  4. liftOver mm39 -> mm10 with /data2st1/junyi/mm39ToMm10.over.chain.gz
  5. bgzip + tabix the mm10 bedGraphs (values 0-100)
  6. copies zero ATAC/RNA bigWigs and cCRE regions
  7. writes manifest.csv / manifest.json

Output:
  /data1st1/junyi/methdata/GSE279860_finetune_input/
    CBM_2_rep1_5mC_mm10.bedgraph.gz (+ .tbi)
    CBM_2_rep1_5hmC_mm10.bedgraph.gz (+ .tbi)
    ... 4 samples
    zero_ATAC_mm10.bw / zero_RNA_mm10.bw
    regions/cCRE_cpg.bed / regions/cpg_test.bed
    manifest.csv / manifest.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
from datetime import datetime

BASE = "/data1st1/junyi/methdata/GSE279860"
CANON = {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY"}
OUT = "/data1st1/junyi/methdata/GSE279860_finetune_input"
GENOME = "/data2st1/wwx/ref_genome_species_raw/mouse/GRCm39_mm39/Mus_musculus.GRCm39.dna.primary_assembly.fa"
CHAIN = "/data2st1/junyi/mm39ToMm10.over.chain.gz"
ZERO = "/data1st1/junyi/methdata/GSE244251/zero_track_mm10.bw"
CCRE = "/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
TESTBED = "/home/junyichen/code/mmllm/output/holdout_testchroms/cpg_test.bed"

SAMPLES = {
    "CBM_2_rep1": "GSM8582102_CBM_2_rep1.masked.bed.gz",
    "CBM_2_rep2": "GSM8582103_CBM_2_rep2.masked.bed.gz",
    "CBM_3_rep1": "GSM8582104_CBM_3_rep1.masked.bed.gz",
    "CBM_3_rep2": "GSM8582105_CBM_3_rep2.masked.bed.gz",
}


def load_chrom_seq(genome: str, chrom: str):
    """Load one mm39 chromosome as bytes via samtools faidx (pyfaidx hangs on this FASTA)."""
    key = chrom[3:] if chrom.startswith("chr") else chrom
    try:
        proc = subprocess.run(["samtools", "faidx", genome, key], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        return None
    data = proc.stdout
    nl = data.find(b"\n")
    if nl < 0:
        raise RuntimeError(f"empty faidx output for {chrom}")
    seq = data[nl + 1:].replace(b"\n", b"").upper()
    return seq


def parse_sample(path: str, chrom_cache: dict):
    """Return (mm39 5mC bedGraph lines, mm39 5hmC bedGraph lines)."""
    tmp_m = []
    tmp_h = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.rstrip().split("\t")
            if len(f) < 12:
                continue
            mark = f[3]
            if mark not in ("m", "h"):
                continue
            if len(f) < 6 or f[5] != "+":
                continue
            chrom = f[0]
            if chrom not in CANON:
                continue
            try:
                start = int(f[1]); end = int(f[2]); value = float(f[10])
            except ValueError:
                continue
            if end <= start:
                continue
            seq = chrom_cache.get(chrom)
            if seq is None:
                seq = load_chrom_seq(GENOME, chrom)
                chrom_cache[chrom] = seq
            if not seq or start + 1 >= len(seq):
                continue
            # CpG forward C only
            if seq[start] != ord("C") or seq[start + 1] != ord("G"):
                continue
            rec = f"{chrom}\t{start}\t{start + 1}\t{value:.6f}\n"
            if mark == "m":
                tmp_m.append(rec)
            else:
                tmp_h.append(rec)
    return tmp_m, tmp_h


def liftover_and_index(src: str, chain: str, out_gz: str, workdir: str, tag: str):
    mapped = os.path.join(workdir, f"{tag}.mapped.bed")
    unmapped = os.path.join(workdir, f"{tag}.unmapped.bed")
    subprocess.run(["liftOver", src, chain, mapped, unmapped], check=True)
    # sort and bgzip
    sorted_bed = os.path.join(workdir, f"{tag}.sorted.bed")
    with open(sorted_bed, "w") as out:
        p1 = subprocess.Popen(["sort", "-k1,1", "-k2,2n", mapped], stdout=out)
        p1.wait()
    subprocess.run(["bgzip", "-f", sorted_bed], check=True)
    bgz = sorted_bed + ".gz"
    if os.path.exists(out_gz):
        os.remove(out_gz)
    os.replace(bgz, out_gz)
    subprocess.run(["tabix", "-p", "bed", "-f", out_gz], check=True)
    n_mapped = sum(1 for _ in open(mapped))
    n_unmapped = 0
    with open(unmapped) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_unmapped += 1
    return n_mapped, n_unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--samples", nargs="+", default=list(SAMPLES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    regions = os.path.join(out, "regions"); os.makedirs(regions, exist_ok=True)
    work = os.path.join(out, "_work"); os.makedirs(work, exist_ok=True)

    # decompress chain once for liftOver
    chain = os.path.join(work, "mm39ToMm10.over.chain")
    if not os.path.exists(chain):
        with gzip.open(CHAIN, "rb") as src, open(chain, "wb") as dst:
            shutil.copyfileobj(src, dst)

    chrom_cache = {}
    manifest = {
        "dataset": "GSE279860", "species": "Mouse", "tissue": "cerebellum",
        "raw_genome": "mm39/GRCm39", "output_genome": "mm10/GRCm38",
        "created": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "note": "CpG forward C only; values are BedMethyl percent (0-100); lifTover mm39->mm10",
        "samples": {},
    }
    rows = []
    for name in args.samples:
        src = os.path.join(BASE, SAMPLES[name])
        if not os.path.exists(src):
            print(f"[skip] missing {src}"); continue
        m_out = os.path.join(out, f"{name}_5mC_mm10.bedgraph.gz")
        h_out = os.path.join(out, f"{name}_5hmC_mm10.bedgraph.gz")
        if not args.force and os.path.exists(m_out) and os.path.exists(h_out):
            print(f"[skip] {name} outputs exist")
            rows.append({"sample": name, "5mC": m_out, "5hmC": h_out})
            continue
        print(f"[parse] {name}")
        rec_m, rec_h = parse_sample(src, chrom_cache)
        print(f"  CpG + rows: 5mC={len(rec_m)} 5hmC={len(rec_h)}")
        for mark, recs, out_gz in (("5mC", rec_m, m_out), ("5hmC", rec_h, h_out)):
            src_bed = os.path.join(work, f"{name}_{mark}.mm39.bed")
            with open(src_bed, "w") as fh:
                fh.writelines(recs)
            print(f"  liftOver {name} {mark} ...")
            nm, nu = liftover_and_index(src_bed, chain, out_gz, work, f"{name}_{mark}")
            print(f"    mapped={nm} unmapped={nu} -> {out_gz}")
        rows.append({"sample": name, "5mC": m_out, "5hmC": h_out})
        manifest["samples"][name] = {"5mC": m_out, "5hmC": h_out}

    for src, name in ((ZERO, "zero_ATAC_mm10.bw"), (ZERO, "zero_RNA_mm10.bw")):
        dst = os.path.join(out, name)
        if args.force or not os.path.exists(dst):
            shutil.copy2(src, dst)
    for src, name in ((CCRE, "cCRE_cpg.bed"), (TESTBED, "cpg_test.bed")):
        dst = os.path.join(regions, name)
        if args.force or not os.path.exists(dst):
            shutil.copy2(src, dst)

    with open(os.path.join(out, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample", "5mC", "5hmC"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
