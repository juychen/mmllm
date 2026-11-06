#!/bin/bash
# Fine-tune the cCRE 2-head model on GSE279860 (mouse cerebellum, Nanopore).
#
# Deliberately a third platform/dataset, prepared 2026-11-02 but never started:
#   data      : GSE279860 CBM_2 / CBM_3 (2 mice x 2 reps), mm39 -> mm10 liftover
#   regions   : cCRE_cpg.bed (80,137) + cpg_test.bed (7,412, same holdout chr8/9/X/Y)
#   ATAC/RNA  : absent in GSE279860 -> zero bigWig + drop-p 1.0
#   init      : SAME checkpoint as GSE244251 / GSE166423 FT runs
#               (2026-10-20-13-31-30 2-head all-others a0.2) so the only change is the dataset
#   epochs    : 30, LR 1e-4, warmup 1
#
# Usage:
#   SAMPLE=CBM_2_rep1 GPU=7 ./finetune_gse279860_2heads.sh
#   SAMPLE=CBM_2_rep1 GPU=7 NUM_EPOCHS=1 SAMPLE_SIZES=64 OUT_DIR=/tmp/gse279860_smoke ./finetune_gse279860_2heads.sh

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

DATA="/data1st1/junyi/methdata/GSE279860_finetune_input"
SAMPLE="${SAMPLE:-CBM_2_rep1}"
M5C="$DATA/${SAMPLE}_5mC_mm10.bedgraph.gz"
HM5C="$DATA/${SAMPLE}_5hmC_mm10.bedgraph.gz"
ATAC="$DATA/zero_ATAC_mm10.bw"
RNA="$DATA/zero_RNA_mm10.bw"
DMR="$DATA/regions/cCRE_cpg.bed"
TESTBED="$DATA/regions/cpg_test.bed"
CKPT="output/AMY_MC/cCRE_cpg/2026-10-20-13-31-30_multiquery_pretrain_ctxall-others_s0.0_a0.2_r0.2_best_80137.pt"
OUT_DIR="${OUT_DIR:-output/GSE279860_finetune/${SAMPLE}_ft30ep}"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="$LOG_DIR/gse279860_${SAMPLE}_finetune_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in "$M5C" "$HM5C" "$ATAC" "$RNA" "$DMR" "$TESTBED" "$CKPT"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

echo "[$(date)] GSE279860 ${SAMPLE} fine-tune | LR=${LR:-1e-4} | epochs=${NUM_EPOCHS:-30} | GPU=${GPU:-7}" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${GPU:-7}" python run_multiquery_pretraining.py \
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
  --mask-neighborhood-bp "${MASK_NEIGHBORHOOD_BP:-50}" \
  --mask-mode cpg_forward \
  --drop-p-5mc 0.0 --drop-p-5hmc 0.0 --drop-p-atac 1.0 --drop-p-rna 1.0 --drop-p-seq 0.0 \
  --scheduler cosine \
  --num-epochs "${NUM_EPOCHS:-30}" \
  --learning-rate "${LR:-1e-4}" \
  --warmup-epochs 1 \
  --warmup-start-factor 0.1 \
  --batch-size "${BATCH_SIZE:-32}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --scheduler-patience 10 \
  --patience 10 \
  --amp \
  --no-export-test-h5ad \
  --init-from-checkpoint "$CKPT" \
  --timestamp "$TS" \
  --scheduler-min-lr 1e-5 \
  --m5c-bedgraph "$M5C" \
  --hm5c-bedgraph "$HM5C" \
  --atac-bw "$ATAC" \
  --rna-coverage-bw "$RNA" \
  --output-csv "${OUT_DIR}/${TS}_gse279860_${SAMPLE}_ft_results.csv" \
  --output-json "${OUT_DIR}/${TS}_gse279860_${SAMPLE}_ft_results.json" \
  --best-checkpoint-path "${OUT_DIR}/${TS}_gse279860_${SAMPLE}_ft_best_{sample_size}.pt" \
  --last-checkpoint-path "${OUT_DIR}/${TS}_gse279860_${SAMPLE}_ft_last_{sample_size}.pt" \
  2>&1 | tee -a "$LOG"

echo "[done] $LOG"
