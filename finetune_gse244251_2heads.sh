#!/bin/bash
# Fine-tune the cCRE 2-head model on GSE244251 (mouse Nanopore, public).
#
# NOTE: this is deliberately separate from the AMY/HIP/PFC brain-region data:
#   data       : GSE244251 Neuron / Astrocyte / Microglia (3 input groups)
#   regions    : cCRE_cpg.bed (same coordinate framework, held-out chr8/9/X/Y)
#   ATAC/RNA   : absent in GSE244251 -> zero bigWig + drop-p 1.0
#   init       : cCRE 2-head a0.2 checkpoint (weights only, new optimizer)
#   epochs     : 30
#   LR         : 1e-4 (smaller than pretrain 1e-3), warmup 1 epoch
#
# Output is under output/GSE244251_finetune/, never under AMY_MC/cCRE_cpg/.
#
# Usage:
#   ./finetune_gse244251_2heads.sh
#   SAMPLE_SIZES=64 NUM_EPOCHS=1 OUT_DIR=/tmp/gse_ft_smoke ./finetune_gse244251_2heads.sh
#   CELLS="Neuron" ./finetune_gse244251_2heads.sh   # per-cell-type model

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

DATA="/data1st1/junyi/methdata/GSE244251/processed_data_tabix_scaled100"
ZERO="/data1st1/junyi/methdata/GSE244251/zero_track_mm10.bw"
GENOME="/data1st2/hannan_25/genome_ref/mm10/Mus_musculus.GRCm38.dna.primary_assembly.fa"
CKPT="output/AMY_MC/cCRE_cpg/2026-10-20-13-31-30_multiquery_pretrain_ctxall-others_s0.0_a0.2_r0.2_best_80137.pt"
DMR="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
TESTBED="output/holdout_testchroms/cpg_test.bed"
OUT_DIR="${OUT_DIR:-output/GSE244251_finetune/cCREregions_3celltypes_ft30ep}"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="$LOG_DIR/gse244251_finetune_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# prepare scaled 0-100 tracks + zero ATAC/RNA bigWig if missing
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  bash prepare_gse244251_finetune_data.sh
fi

CELLS="${CELLS:-Neuron Astrocyte Microglia}"
cells=($CELLS)
m5c_paths=(); hm5c_paths=(); atac_paths=(); rna_paths=()
for c in "${cells[@]}"; do
  m5c_paths+=("$DATA/5mC_${c}_5mC.bedgraph.gz")
  hm5c_paths+=("$DATA/5hmC_${c}_5hmC.bedgraph.gz")
  atac_paths+=("$ZERO")
  rna_paths+=("$ZERO")
done

for f in "${m5c_paths[@]}" "${hm5c_paths[@]}" "$ZERO" "$CKPT" "$DMR" "$TESTBED"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

echo "[$(date)] GSE244251 fine-tune | cells=${cells[*]} | LR=${LR:-1e-4} | epochs=${NUM_EPOCHS:-30} | drop ATAC/RNA=1.0" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${GPU:-0}" python run_multiquery_pretraining.py \
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
  --drop-p-5mc 0.0 --drop-p-5hmc 0.0 --drop-p-atac 1.0 --drop-p-rna 1.0 --drop-p-seq 0.0 \
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
  --m5c-bedgraph "${m5c_paths[@]}" \
  --hm5c-bedgraph "${hm5c_paths[@]}" \
  --atac-bw "${atac_paths[@]}" \
  --rna-coverage-bw "${rna_paths[@]}" \
  --output-csv "${OUT_DIR}/${TS}_gse244251_2heads_ft_results.csv" \
  --output-json "${OUT_DIR}/${TS}_gse244251_2heads_ft_results.json" \
  --best-checkpoint-path "${OUT_DIR}/${TS}_gse244251_2heads_ft_best_{sample_size}.pt" \
  --last-checkpoint-path "${OUT_DIR}/${TS}_gse244251_2heads_ft_last_{sample_size}.pt" \
  2>&1 | tee -a "$LOG"

echo "[done] $LOG"
