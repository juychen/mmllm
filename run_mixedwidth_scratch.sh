#!/bin/bash
# Mixed-width pretraining FROM SCRATCH (100 epochs, single GPU).
#
# Same recipe as the production 2-head bp50 run
#   (2026-11-03-13-17-57_..._cpgpair_bp50_lossc_drop0.1_100ep)
# except that every selected CpG pair draws its own hole width from WIDTHS
# instead of always using +/-50 bp.  Validation keeps the fixed +/-50 bp mask
# so the val column stays comparable with the other runs; the 6-width probe
# (probe_mask_regimes.py) then tells us whether the model is protocol-robust
# rather than protocol-locked.
#
# Usage:
#   GPU=0 ./run_mixedwidth_scratch.sh
#   GPU=0 NUM_EPOCHS=1 SAMPLE_SIZES=64 OUT_DIR=/tmp/mix_scratch_smoke ./run_mixedwidth_scratch.sh

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

M5C="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz"
HM5C="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"
ATAC="/data1st1/junyi/methdata/atac/AMY_MC_track.bw"
RNA="/data2st1/junyi/output/sn0615/BULK_AMY/AMY_MC.bw"
DMR="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
TESTBED="output/holdout_testchroms/cpg_test.bed"

OUT_DIR="${OUT_DIR:-output/AMY_MC/cCRE_cpg_mixedwidth/scratch100ep}"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="$LOG_DIR/mixedwidth_scratch_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in "$M5C" "$HM5C" "$ATAC" "$RNA" "$DMR" "$TESTBED"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

echo "[$(date)] mixed-width SCRATCH | widths=${WIDTHS:-0,10,25,50,100} | lr=${LR:-1e-3} | epochs=${NUM_EPOCHS:-100} | GPU=${GPU:-0}" | tee -a "$LOG"
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
  --mask-neighborhood-bp 50 \
  --mask-widths "${WIDTHS:-0,10,25,50,100}" \
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
  --output-csv "${OUT_DIR}/${TS}_mixedwidth_scratch_results.csv" \
  --output-json "${OUT_DIR}/${TS}_mixedwidth_scratch_results.json" \
  --best-checkpoint-path "${OUT_DIR}/${TS}_mixedwidth_scratch_best_{sample_size}.pt" \
  --last-checkpoint-path "${OUT_DIR}/${TS}_mixedwidth_scratch_last_{sample_size}.pt" \
  2>&1 | tee -a "$LOG"

echo "[done] $LOG"
