#!/bin/bash
# Prepare GSE244251 tracks for fine-tuning (separate from the AMY/HIP/PFC
# brain-region data):
#   1) 5mC/5hmC bedGraph values are 0-1; scale to 0-100 to match pretraining.
#   2) create a zero ATAC/RNA bigWig on mm10 (GSE244251 has no ATAC/RNA).
#
# Inputs : /data1st1/junyi/methdata/GSE244251/processed_data_tabix
# Outputs: /data1st1/junyi/methdata/GSE244251/processed_data_tabix_scaled100
#          /data1st1/junyi/methdata/GSE244251/zero_track_mm10.bw

set -euo pipefail

IN="/data1st1/junyi/methdata/GSE244251/processed_data_tabix"
OUT="/data1st1/junyi/methdata/GSE244251/processed_data_tabix_scaled100"
GENOME="/data1st2/hannan_25/genome_ref/mm10/Mus_musculus.GRCm38.dna.primary_assembly.fa"
ZERO="/data1st1/junyi/methdata/GSE244251/zero_track_mm10.bw"
mkdir -p "$OUT"

for cell in Neuron Astrocyte Microglia; do
  for mark in 5mC 5hmC; do
    src="$IN/${mark}_${cell}_${mark}.bedgraph.gz"
    dst="$OUT/${mark}_${cell}_${mark}.bedgraph.gz"
    if [[ -s "$dst" && -s "$dst.tbi" ]]; then
      echo "[skip] $dst"
      continue
    fi
    echo "[scale] $src -> $dst (x100)"
    LC_ALL=C zcat "$src" | LC_ALL=C awk 'BEGIN{OFS="\t"} {print $1,$2,$3,$4*100}' | bgzip -c > "$dst"
    tabix -p bed -f "$dst"
  done
done

if [[ ! -s "$ZERO" ]]; then
  echo "[zero] $ZERO"
  python - "$GENOME" "$ZERO" <<'PY'
import sys
import pyBigWig
import pyfaidx

genome_path, out_path = sys.argv[1], sys.argv[2]
g = pyfaidx.Fasta(genome_path, as_raw=True)
chroms = [(c if c.startswith("chr") else "chr" + c, len(g[c])) for c in g.keys()]
bw = pyBigWig.open(out_path, "w")
bw.addHeader(chroms, maxZooms=0)
for c, length in chroms:
    bw.addEntries([c], [0], ends=[length], values=[0.0])
bw.close()
print("wrote", out_path, "chroms", len(chroms))
PY
else
  echo "[skip] $ZERO"
fi

echo "[done] scaled tracks + zero bigWig"
