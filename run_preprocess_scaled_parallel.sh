#!/bin/bash
# Three parallel workers; each handles 2 groups (ATAC + RNA).
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

OUT="/data1st1/junyi/methdata/scaled_tracks"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOG_DIR"

run_worker() {
  local tag="$1"; shift
  python preprocess_log1p_p99_scaled.py \
    --out-dir "$OUT" \
    --groups "$@" \
    --tracks atac rna \
    --percentile 99 \
    --scale-to-one \
    --tag "$tag" \
    > "${LOG_DIR}/preprocess_scaled_${tag}_${TS}.out" 2>&1
}

run_worker w1 AMY_MC AMY_MW & P1=$!
run_worker w2 HIP_MC HIP_MW & P2=$!
run_worker w3 PFC_MC PFC_MW & P3=$!

wait "$P1"; C1=$?
wait "$P2"; C2=$?
wait "$P3"; C3=$?

echo "[$(date)] done: w1=$C1 w2=$C2 w3=$C3"
exit $(( C1 + C2 + C3 ))
