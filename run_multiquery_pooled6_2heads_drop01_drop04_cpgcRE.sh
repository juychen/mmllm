#!/bin/bash
# Pooled 6-group, 2-head cCRE CpG training with TWO modality-dropout settings:
#   ATAC/RNA dropout = 0.1  and  0.4
#
# This is the same model/protocol as
#   run_multiquery_pooled6_2heads_drop02_cpgcRE.sh
# (2 heads: 5mC + 5hmC; context ATAC + RNA + sequence, all-others;
#  cCRE_cpg.bed; 6 groups = AMY/HIP/PFC x MC/MW; 30 epochs; batch 32 x accum 8)
#
# Differences:
#   - two runs: drop 0.1 and drop 0.4
#   - all paths are overridable, so it can run on 10.21.20.32
#   - default is SEQUENTIAL on one GPU; set PARALLEL=1 GPUS="0 1" to run both
#     variants at the same time.
#
# This script is written but NOT submitted here. Run it on 10.21.20.32.
#
# ---------------------------------------------------------------------------
# Usage on 10.21.20.32
# ---------------------------------------------------------------------------
#   cd <repo>
#   chmod +x run_multiquery_pooled6_2heads_drop01_drop04_cpgcRE.sh
#
#   # sequential, single GPU
#   GPU=0 ./run_multiquery_pooled6_2heads_drop01_drop04_cpgcRE.sh
#
#   # parallel, two GPUs
#   PARALLEL=1 GPUS="0 1" ./run_multiquery_pooled6_2heads_drop01_drop04_cpgcRE.sh
#
#   # smoke test (64 x 6 rows, 1 epoch)
#   N_SMOKE=1 GPU=0 ./run_multiquery_pooled6_2heads_drop01_drop04_cpgcRE.sh
#
# If the data layout differs on 10.21.20.32, override:
#   METH_DIR=... ATAC_DIR=... RNA_DIR=... DMR_CSV=... OUT_ROOT=... \
#   CONDA_ENV=... CONDA_SH=... ./run_multiquery_pooled6_2heads_drop01_drop04_cpgcRE.sh
#
# Runtime per run: 480,822 windows/epoch; ~4.5-5.5 h/epoch; 30 epochs ~5-6 days.
# Two runs sequential ~10-12 days; parallel on two GPUs ~5-6 days.
# ---------------------------------------------------------------------------

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMLLM_DIR="${MMLLM_DIR:-$SCRIPT_DIR}"
cd "$MMLLM_DIR" || { echo "[error] cannot cd to $MMLLM_DIR" >&2; exit 1; }
export PYTHONUNBUFFERED=1

# ---- conda -----------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-evo2}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[error] conda.sh not found; set CONDA_SH=/path/to/conda.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV" || { echo "[error] conda activate $CONDA_ENV failed" >&2; exit 1; }

# ---- data roots (override on 10.21.20.32 if needed) -------------------------
METH_DIR="${METH_DIR:-/data2st1/junyi/output/llm0401/processed_meth}"
ATAC_DIR="${ATAC_DIR:-/data1st1/junyi/methdata/atac}"
RNA_DIR="${RNA_DIR:-/data2st1/junyi/output/sn0615}"
DMR_CSV="${DMR_CSV:-/data2st1/junyi/generegion_vM23/cCRE_cpg.bed}"
OUT_ROOT="${OUT_ROOT:-output/ALL_GROUPS/cCRE_cpg}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"

# ---- run settings -----------------------------------------------------------
DROPS=(${DROPS:-0.1 0.4})
GPUS=(${GPUS:-0 1})
PARALLEL="${PARALLEL:-0}"          # 0 = sequential, 1 = one process per drop
GPU="${GPU:-${GPUS[0]}}"           # used when PARALLEL=0
NUM_EPOCHS="${NUM_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLE_SIZES="${SAMPLE_SIZES:-all}"
RUN_TAG="${RUN_TAG:-}"

# ---- pooled groups, fixed order --------------------------------------------
regions=(AMY AMY HIP HIP PFC PFC)
conditions=(MC MW MC MW MC MW)
m5c_paths=(); hm5c_paths=(); atac_paths=(); rna_paths=()
for i in "${!regions[@]}"; do
  r="${regions[$i]}"; c="${conditions[$i]}"
  m5c_paths+=("$METH_DIR/${c}_${r}.CG.m.bedGraph.gz")
  hm5c_paths+=("$METH_DIR/${c}_${r}.CG.h.bedGraph.gz")
  atac_paths+=("$ATAC_DIR/${r}_${c}_track.bw")
  rna_paths+=("$RNA_DIR/BULK_${r}/${r}_${c}.bw")
done

for f in "${m5c_paths[@]}" "${hm5c_paths[@]}" "${atac_paths[@]}" "${rna_paths[@]}" "$DMR_CSV"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

# ---- smoke mode -------------------------------------------------------------
if [[ -n "${N_SMOKE:-}" ]]; then
  NUM_EPOCHS=1
  BATCH_SIZE=16
  GRAD_ACCUM=1
  NUM_WORKERS=0
  SAMPLE_SIZES=64
  OUT_ROOT="/tmp/mq_pooled6_drop_smoke"
  LOG_DIR="/tmp"
  echo "[smoke] 64 rows x 6 groups, 1 epoch, batch 16, drops=${DROPS[*]}"
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR"

run_one() {
  local drop="$1"
  local gpu="$2"
  local ts
  ts="$(date +%Y-%m-%d-%H-%M-%S)"
  local tag
  tag="$(printf '%s' "$drop" | tr -d '.')"          # 0.1 -> 01, 0.4 -> 04
  local run_label="multiquery_pooled6_2heads_s0.0_a${drop}_r${drop}${RUN_TAG:+_${RUN_TAG}}"
  local log_file="${LOG_DIR}/${ts}_${run_label}.out"

  echo "[$(date)] drop=${drop} | GPU=${gpu} | epochs=${NUM_EPOCHS}" | tee -a "$log_file"
  echo "  groups : ${regions[*]} x ${conditions[*]} (6)" | tee -a "$log_file"
  echo "  heads  : 5mc 5hmc; context ATAC+RNA+seq, all-others" | tee -a "$log_file"
  echo "  mask   : 15% cpg_forward; ATAC/RNA drop=${drop}; 5mc/5hmc/seq=0" | tee -a "$log_file"
  echo "  regions: $DMR_CSV" | tee -a "$log_file"
  echo "  output : $OUT_ROOT" | tee -a "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" python run_multiquery_pretraining.py \
    --sample-sizes "$SAMPLE_SIZES" \
    --use-all-input-groups \
    --dmr-csv "$DMR_CSV" \
    --target-length 16384 \
    --num-blocks 2 \
    --hidden-dim 64 \
    --fusion-type cross_hyena \
    --query-tracks 5mc 5hmc \
    --context-mode all-others \
    --mask-fraction 0.15 \
    --mask-mode cpg_forward \
    --augment-reverse-complement \
    --drop-p-5mc 0.0 --drop-p-5hmc 0.0 \
    --drop-p-atac "$drop" --drop-p-rna "$drop" --drop-p-seq 0.0 \
    --scheduler cosine \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-workers "$NUM_WORKERS" \
    --scheduler-patience 15 \
    --patience 15 \
    --amp \
    --timestamp "$ts" \
    --scheduler-min-lr 1e-5 \
    --m5c-bedgraph "${m5c_paths[@]}" \
    --hm5c-bedgraph "${hm5c_paths[@]}" \
    --atac-bw "${atac_paths[@]}" \
    --rna-coverage-bw "${rna_paths[@]}" \
    --output-csv "${OUT_ROOT}/${ts}_${run_label}_results.csv" \
    --output-json "${OUT_ROOT}/${ts}_${run_label}_results.json" \
    --best-checkpoint-path "${OUT_ROOT}/${ts}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${OUT_ROOT}/${ts}_${run_label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local code=${PIPESTATUS[0]}
  if [[ $code -eq 0 ]]; then
    echo "[$(date)] drop=${drop} finished" | tee -a "$log_file"
  else
    echo "[$(date)] drop=${drop} FAILED (exit $code)" | tee -a "$log_file"
  fi
  return "$code"
}

failed=0
if [[ "$PARALLEL" == "1" ]]; then
  echo "[mode] parallel; drops=${DROPS[*]}; GPUs=${GPUS[*]}"
  pids=()
  for i in "${!DROPS[@]}"; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    run_one "${DROPS[$i]}" "$gpu" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || failed=$((failed + 1))
  done
else
  echo "[mode] sequential; drops=${DROPS[*]}; GPU=${GPU}"
  for drop in "${DROPS[@]}"; do
    run_one "$drop" "$GPU" || failed=$((failed + 1))
  done
fi

if [[ $failed -eq 0 ]]; then
  echo "[$(date)] all runs finished"
else
  echo "[$(date)] ${failed} run(s) failed"
fi
exit "$failed"
