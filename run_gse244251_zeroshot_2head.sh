#!/bin/bash
# Zero-shot evaluation of a trained 2-head model on GSE244251 mouse Nanopore
# 5mC/5hmC.  ATAC/RNA are absent in this dataset -> drop rate 100%.
# 15% shared CG mask; only CG sites with both 5mC and 5hmC coverage.
# Default REGIONS_BED = cCRE held-out test windows (chr8/9/X/Y, 7,412).
#
#   CKPT        : default = cCRE 2-head a0.2
#   MAX_WINDOWS : per cell type (default 300)
#   CELLS       : default "Neuron Astrocyte Microglia"
#   GPU         : default 0

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

CKPT="${CKPT:-output/AMY_MC/cCRE_cpg/2026-10-20-13-31-30_multiquery_pretrain_ctxall-others_s0.0_a0.2_r0.2_best_80137.pt}"
DATA="/data1st1/junyi/methdata/GSE244251/processed_data_tabix"
GENOME="/data1st2/hannan_25/genome_ref/mm10/Mus_musculus.GRCm38.dna.primary_assembly.fa"
OUT_DIR="${OUT_DIR:-output/GSE244251_zeroshot}"
REGIONS_BED="${REGIONS_BED:-output/holdout_testchroms/cpg_test.bed}"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="$LOG_DIR/gse244251_zeroshot_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

[[ -s "$CKPT" ]] || { echo "[error] missing checkpoint: $CKPT" >&2; exit 1; }
[[ -d "$DATA" ]] || { echo "[error] missing data dir: $DATA" >&2; exit 1; }
[[ -s "$GENOME" ]] || { echo "[error] missing genome: $GENOME" >&2; exit 1; }

CUDA_VISIBLE_DEVICES="${GPU:-0}" python eval_gse244251_zeroshot.py \
  --checkpoint "$CKPT" \
  --data-dir "$DATA" \
  --genome "$GENOME" \
  --out-dir "$OUT_DIR" \
  --cell-types ${CELLS:-Neuron Astrocyte Microglia} \
  --regions-bed "$REGIONS_BED" \
  --max-windows "${MAX_WINDOWS:-300}" \
  --min-cg "${MIN_CG:-20}" \
  --mask-fraction "${MASK_FRACTION:-0.15}" \
  --amp \
  2>&1 | tee "$LOG"

echo "[done] $LOG"
