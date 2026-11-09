#!/bin/bash
# E6: contiguous-block masking (min 100 bp), 15% of CpGs as anchors.
#
# Every sampled CpG opens a contiguous hidden block (adaptive length, clamped to
# [MIN_BP, MAX_BP]); the loss is computed on every forward C inside the block.
# Evaluation keeps the fixed +/-50 bp mask so the val column stays comparable.
#
# Usage:
#   GPU=2 ./run_block_ft.sh
#   GPU=2 NUM_EPOCHS=1 SAMPLE_SIZES=64 OUT_DIR=/tmp/block_smoke ./run_block_ft.sh

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

M5C="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz"
HM5C="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"
ATAC="/data1st1/junyi/methdata/atac/AMY_MC_track.bw"
RNA="/data2st1/junyi/output/sn0615/BULK_AMY/AMY_MC.bw"
DMR="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
TESTBED="output/holdout_testchroms/cpg_test.bed"
CKPT="output/AMY_MC/cCRE_cpg/2026-11-03-13-17-57_multiquery_2heads_cpgpair_bp50_lossc_drop0.1_100ep_best_80137.pt"
OUT_DIR="${OUT_DIR:-output/AMY_MC/cCRE_cpg_blockmask/ft5ep}"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="$LOG_DIR/block_ft_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in "$M5C" "$HM5C" "$ATAC" "$RNA" "$DMR" "$TESTBED" "$CKPT"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

echo "[$(date)] block-mask FT | min=${MIN_BP:-100} max=${MAX_BP:-400} | lr=${LR:-2e-4} | epochs=${NUM_EPOCHS:-5} | GPU=${GPU:-6}" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${GPU:-6}" python run_multiquery_pretraining.py \
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
  --mask-neighborhood-bp 50 \
  --mask-block --mask-block-min-bp "${MIN_BP:-100}" --mask-block-max-bp "${MAX_BP:-400}" \
  --mask-mode cpg_forward \
  --drop-p-5mc 0.0 --drop-p-5hmc 0.0 --drop-p-atac 0.1 --drop-p-rna 0.1 --drop-p-seq 0.0 \
  --scheduler cosine \
  --num-epochs "${NUM_EPOCHS:-5}" \
  --learning-rate "${LR:-2e-4}" \
  --warmup-epochs 0 \
  --batch-size "${BATCH_SIZE:-32}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --scheduler-patience 10 \
  --patience 10 \
  --amp \
  --augment-reverse-complement \
  --no-export-test-h5ad \
  --init-from-checkpoint "$CKPT" \
  --timestamp "$TS" \
  --scheduler-min-lr 1e-5 \
  --m5c-bedgraph "$M5C" \
  --hm5c-bedgraph "$HM5C" \
  --atac-bw "$ATAC" \
  --rna-coverage-bw "$RNA" \
  --output-csv "${OUT_DIR}/${TS}_block_ft_results.csv" \
  --output-json "${OUT_DIR}/${TS}_block_ft_results.json" \
  --best-checkpoint-path "${OUT_DIR}/${TS}_block_ft_best_{sample_size}.pt" \
  --last-checkpoint-path "${OUT_DIR}/${TS}_block_ft_last_{sample_size}.pt" \
  2>&1 | tee -a "$LOG"

echo "[done] $LOG"
