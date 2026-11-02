#!/bin/bash
# Fine-tune cCRE 2-head a0.2 on GSE166423 **Adult Purkinje** only.
# Inputs are the preprocessed bigWigs in GSE166423_finetune_input:
#   5mC/5hmC scaled 0-100; real ATAC; zero RNA.
#
# 30 epochs, LR 1e-4, warmup 1, ATAC dropout 0.2 (present), RNA dropout 1.0.
# Separate output: output/GSE166423_finetune/Adult_ft30ep

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

DATA="/data1st1/junyi/methdata/GSE166423_finetune_input"
M5C="$DATA/Purkinje_Adult_5mC_scaled100.bw"
HM5C="$DATA/Purkinje_Adult_5hmC_scaled100.bw"
ATAC="$DATA/Purkinje_Adult_ATAC.bw"
RNA="$DATA/zero_RNA_mm10.bw"
DMR="$DATA/regions/cCRE_cpg.bed"
TESTBED="$DATA/regions/cpg_test.bed"
CKPT="output/AMY_MC/cCRE_cpg/2026-10-20-13-31-30_multiquery_pretrain_ctxall-others_s0.0_a0.2_r0.2_best_80137.pt"
OUT_DIR="${OUT_DIR:-output/GSE166423_finetune/Adult_ft30ep}"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="$LOG_DIR/gse166423_adult_finetune_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in "$M5C" "$HM5C" "$ATAC" "$RNA" "$DMR" "$TESTBED" "$CKPT"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

echo "[$(date)] GSE166423 Adult Purkinje fine-tune | LR=${LR:-1e-4} | epochs=${NUM_EPOCHS:-30} | ATAC drop=${DROP_ATAC:-0.2} | RNA drop=1.0" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${GPU:-3}" python run_multiquery_pretraining.py \
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
  --mask-mode cpg_forward \
  --drop-p-5mc 0.0 --drop-p-5hmc 0.0 --drop-p-atac "${DROP_ATAC:-0.2}" --drop-p-rna 1.0 --drop-p-seq 0.0 \
  --scheduler cosine \
  --num-epochs "${NUM_EPOCHS:-30}" \
  --learning-rate "${LR:-1e-4}" \
  --warmup-epochs "${WARMUP_EPOCHS:-1}" \
  --warmup-start-factor 0.1 \
  --batch-size "${BATCH_SIZE:-32}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --scheduler-patience "${SCHED_PATIENCE:-10}" \
  --patience "${PATIENCE:-10}" \
  --amp \
  --no-export-test-h5ad \
  --init-from-checkpoint "$CKPT" \
  --timestamp "$TS" \
  --scheduler-min-lr 1e-5 \
  --m5c-bedgraph "$M5C" \
  --hm5c-bedgraph "$HM5C" \
  --atac-bw "$ATAC" \
  --rna-coverage-bw "$RNA" \
  --output-csv "${OUT_DIR}/${TS}_gse166423_adult_ft_results.csv" \
  --output-json "${OUT_DIR}/${TS}_gse166423_adult_ft_results.json" \
  --best-checkpoint-path "${OUT_DIR}/${TS}_gse166423_adult_ft_best_{sample_size}.pt" \
  --last-checkpoint-path "${OUT_DIR}/${TS}_gse166423_adult_ft_last_{sample_size}.pt" \
  2>&1 | tee -a "$LOG"

echo "[done] $LOG"
