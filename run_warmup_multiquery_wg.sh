#!/bin/bash
# Warm-start the multi-query masked pretraining model:
#   weights  : the cCRE checkpoint from run_all_allquery_cpg.sh
#              (logs/run_all_allquery_cpg.sh.2026-10-13-17-10-52.out)
#   data     : whole-genome 16kb windows + a fixed cpg_test.bed holdout
#              (same data as run_all_multiquery_wg.sh)
#   schedule : smaller LR (3e-4 vs 1e-3) with a 3-epoch linear warm-up, then cosine
#   optimizer / scheduler / epoch are reset — this is a warm start, not a resume
#
# Usage:
#   ./run_warmup_multiquery_wg.sh [fusion_type] [seq_drop_p] [atac_drop_p] [rna_drop_p]
#   GPUS="4 5" ./run_warmup_multiquery_wg.sh    # pin GPUs (default: auto-pick idle ones)
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

# ---- hyper-parameters --------------------------------------------------------
fusion_type="${1:-cross_hyena}"
seq_drop_p="${2:-0.0}"
atac_drop_p="${3:-0.0}"
rna_drop_p="${4:-0.0}"

INIT_CKPT="/data1st1/junyi/output/mmllm/AMY_MC/cCRE_cpg/2026-10-13-17-10-53_multiquery_pretrain_ctxall-others_s0.0_a0.0_r0.0_best_80137.pt"
LEARNING_RATE=3e-4
SCHEDULER_MIN_LR=1e-6
WARMUP_EPOCHS=3
WARMUP_START_FACTOR=0.1
NUM_EPOCHS=40
PATIENCE=0                 # 0 = no early stop, let the cosine schedule finish
MASK_FRACTION=0.15
BATCH_SIZE=32
GRAD_ACCUM=8
NUM_WORKERS=8
REGION=AMY
CONDITION=MC

# ---- data --------------------------------------------------------------------
DMR_CSV="/data1st1/junyi/output/mmllm/holdout_testchroms/wg_trainpool_plus_cpgtest.bed"
TEST_BED="output/holdout_testchroms/cpg_test.bed"
M5C_BEDGRAPH="/data2st1/junyi/output/llm0401/processed_meth/${CONDITION}_${REGION}.CG.m.bedGraph.gz"
HM5C_BEDGRAPH="/data2st1/junyi/output/llm0401/processed_meth/${CONDITION}_${REGION}.CG.h.bedGraph.gz"
ATAC_BW="/data1st1/junyi/methdata/atac/${REGION}_${CONDITION}_track.bw"
RNA_BW="/data2st1/junyi/output/sn0615/BULK_${REGION}/${REGION}_${CONDITION}.bw"

# ---- GPU (auto unless GPUS is set) -------------------------------------------
if [[ -n "${GPUS:-}" ]]; then
  gpus=($GPUS)
else
  mapfile -t gpus < <(
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' ' \
      | awk -F',' '$2+0 <= 20000 && $3+0 >= 40000 { print $1 }' \
      | sort -n
  )
  if [[ ${#gpus[@]} -eq 0 ]]; then
    echo "ERROR: no idle GPU (need used <= 20000 MiB and free >= 40000 MiB)." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >&2
    exit 1
  fi
fi

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: init checkpoint not found: $INIT_CKPT" >&2
  exit 1
fi

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
output_dir="output/${REGION}_${CONDITION}/wg_trainpool_plus_cpgtest_warmup_from_cpg1013"
mkdir -p "$output_dir"
run_label="multiquery_warmup_ctxall-others_s${seq_drop_p}_a${atac_drop_p}_r${rna_drop_p}_lr${LEARNING_RATE}"
log_file="${output_dir}/${current_time}_${run_label}.log"

echo "[$(date)] [${REGION}_${CONDITION}] GPU${gpus[0]} | warm-start from $(basename "$INIT_CKPT")"
echo "[$(date)] [${REGION}_${CONDITION}] lr=${LEARNING_RATE} (min ${SCHEDULER_MIN_LR}) | warmup=${WARMUP_EPOCHS}ep | epochs=${NUM_EPOCHS} | drops: s=${seq_drop_p} a=${atac_drop_p} r=${rna_drop_p}"

CUDA_VISIBLE_DEVICES="${gpus[0]}" python run_multiquery_pretraining.py \
  --sample-sizes all \
  --dmr-csv "$DMR_CSV" \
  --test-bed "$TEST_BED" \
  --target-length 16384 \
  --num-blocks 2 \
  --hidden-dim 64 \
  --fusion-type "$fusion_type" \
  --query-tracks 5mc 5hmc \
  --context-mode all-others \
  --mask-fraction "$MASK_FRACTION" \
  --mask-mode cpg_forward \
  --drop-p-seq "$seq_drop_p" \
  --drop-p-atac "$atac_drop_p" \
  --drop-p-rna "$rna_drop_p" \
  --augment-reverse-complement \
  --init-from-checkpoint "$INIT_CKPT" \
  --scheduler cosine \
  --learning-rate "$LEARNING_RATE" \
  --scheduler-min-lr "$SCHEDULER_MIN_LR" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --warmup-start-factor "$WARMUP_START_FACTOR" \
  --num-epochs "$NUM_EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --num-workers "$NUM_WORKERS" \
  --amp \
  --seed 7 \
  --timestamp "$current_time" \
  --m5c-bedgraph "$M5C_BEDGRAPH" \
  --hm5c-bedgraph "$HM5C_BEDGRAPH" \
  --atac-bw "$ATAC_BW" \
  --rna-coverage-bw "$RNA_BW" \
  --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
  --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
  --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
  --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
  2>&1 | tee -a "$log_file"

echo "[$(date)] [${REGION}_${CONDITION}] done — log: ${log_file}"
