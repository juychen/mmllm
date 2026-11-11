#!/bin/bash
# Per-region 2-head pretraining, buffer-0 masking, from scratch.
# Usage: REGION=AMY_MW GPU=0 ./run_region_bp0_scratch.sh
set -e
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

REGION="${REGION:?set REGION, e.g. AMY_MW}"
BRAIN="${REGION%%_*}"
TAG="${REGION##*_}"
M5C="/data2st1/junyi/output/llm0401/processed_meth/${TAG}_${BRAIN}.CG.m.bedGraph.gz"
HM5C="/data2st1/junyi/output/llm0401/processed_meth/${TAG}_${BRAIN}.CG.h.bedGraph.gz"
ATAC="/data1st1/junyi/methdata/atac/${BRAIN}_${TAG}_track.bw"
RNA="/data2st1/junyi/output/sn0615/BULK_${BRAIN}/${REGION}.bw"
DMR="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
TESTBED="output/holdout_testchroms/cpg_test.bed"

TS="$(date +%Y-%m-%d-%H-%M-%S)"
OUT_DIR="${OUT_DIR:-output/regions/${REGION}/cCRE_cpg_bp0/scratch100ep}"
LOG_DIR="/home/junyichen/logs"
LOG="$LOG_DIR/region_bp0_${REGION}_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in "$M5C" "$HM5C" "$ATAC" "$RNA" "$DMR" "$TESTBED"; do
  [[ -s "$f" ]] || { echo "[error] missing: $f" >&2; exit 1; }
done

echo "[$(date)] REGION=${REGION} bp0 drop0.1 2heads epochs=${NUM_EPOCHS:-100} GPU=${GPU:-0}" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u run_multiquery_pretraining.py \
  --sample-sizes "${SAMPLE_SIZES:-all}" \
  --use-all-input-groups \
  --dmr-csv "$DMR" \
  --test-bed "$TESTBED" \
  --target-length 16384 \
  --num-blocks 2 \
  --hidden-dim 64 \
  --fusion-type cross_hyena \
  --query-tracks 5mc 5hmc \
  --context-mode all-others \
  --mask-fraction 0.15 \
  --mask-neighborhood-bp 0 \
  --mask-mode cpg_forward \
  --drop-p-5mc 0.0 --drop-p-5hmc 0.0 --drop-p-atac 0.1 --drop-p-rna 0.1 --drop-p-seq 0.0 \
  --scheduler cosine \
  --num-epochs "${NUM_EPOCHS:-100}" \
  --learning-rate "${LR:-1e-3}" \
  --warmup-epochs 0 \
  --batch-size "${BATCH_SIZE:-32}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --scheduler-patience 10 \
  --patience 10 \
  --amp \
  --augment-reverse-complement \
  --no-export-test-h5ad \
  --timestamp "$TS" \
  --scheduler-min-lr 1e-5 \
  --m5c-bedgraph "$M5C" \
  --hm5c-bedgraph "$HM5C" \
  --atac-bw "$ATAC" \
  --rna-coverage-bw "$RNA" \
  --output-csv "${OUT_DIR}/${TS}_results.csv" \
  --output-json "${OUT_DIR}/${TS}_results.json" \
  --best-checkpoint-path "${OUT_DIR}/${TS}_best_{sample_size}.pt" \
  --last-checkpoint-path "${OUT_DIR}/${TS}_last_{sample_size}.pt" \
  2>&1 | tee -a "$LOG"

echo "[done] $LOG"
