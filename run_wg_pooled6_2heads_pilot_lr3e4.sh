#!/bin/bash
# WG pooled 6-group, 2 heads, 5-epoch pilot at a lower LR + warmup.
# Purpose: check whether dropout 0.0 and 0.2 stop collapsing at LR 1e-3.
#
#   LR 3e-4, warmup 2 epochs, cosine, 5 epochs
#   drops: 0.0 and 0.2 (d01 0.1 is already running at LR 1e-3)
#   GPUs : 0 and 2 (GPU1 is d01; GPU5 is the 4-head 100-epoch run)

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

DMR="/data1st1/junyi/output/mmllm/holdout_testchroms/wg_trainpool_plus_cpgtest.bed"
TESTBED="/home/junyichen/code/mmllm/output/holdout_testchroms/cpg_test.bed"
OUT_ROOT="output/ALL_GROUPS_WG/wg_pooled6_2heads_pilot_lr3e4"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

M="/data2st1/junyi/output/llm0401/processed_meth"
A="/data1st1/junyi/methdata/atac"
R="/data2st1/junyi/output/sn0615"

regions=(AMY AMY HIP HIP PFC PFC)
conditions=(MC MW MC MW MC MW)
m5c_paths=(); hm5c_paths=(); atac_paths=(); rna_paths=()
for i in "${!regions[@]}"; do
  r="${regions[$i]}"; c="${conditions[$i]}"
  m5c_paths+=("$M/${c}_${r}.CG.m.bedGraph.gz")
  hm5c_paths+=("$M/${c}_${r}.CG.h.bedGraph.gz")
  atac_paths+=("$A/${r}_${c}_track.bw")
  rna_paths+=("$R/BULK_${r}/${r}_${c}.bw")
done

for f in "${m5c_paths[@]}" "${hm5c_paths[@]}" "${atac_paths[@]}" "${rna_paths[@]}" "$DMR" "$TESTBED"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

gpus=(${GPUS:-0 2})
drops=(${DROPS:-0.0 0.2})
num_epochs="${NUM_EPOCHS:-5}"
lr="${LR:-3e-4}"
warmup="${WARMUP_EPOCHS:-2}"
seed="${SEED:-7}"
batch_size="${BATCH_SIZE:-32}"
grad_accum="${GRAD_ACCUM:-8}"
num_workers="${NUM_WORKERS:-8}"

run_one() {
  local tag="$1"; local drop="$2"; local gpu="$3"
  local log="$LOG_DIR/wg_pooled6_2head_pilot_${tag}_${TS}.out"
  local label="wg_pooled6_2heads_s0.0_a${drop}_r${drop}_lr${lr}_wup${warmup}_${num_epochs}ep"
  echo "[$(date)] ${tag} | GPU ${gpu} | drop=${drop} | lr=${lr} warmup=${warmup} epochs=${num_epochs}" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" python run_multiquery_pretraining.py \
    --sample-sizes all \
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
    --augment-reverse-complement \
    --drop-p-5mc 0.0 --drop-p-5hmc 0.0 --drop-p-atac "$drop" --drop-p-rna "$drop" --drop-p-seq 0.0 \
    --scheduler cosine \
    --num-epochs "$num_epochs" \
    --learning-rate "$lr" \
    --warmup-epochs "$warmup" \
    --warmup-start-factor 0.1 \
    --seed "$seed" \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$grad_accum" \
    --num-workers "$num_workers" \
    --scheduler-patience 15 \
    --patience 15 \
    --amp \
    --no-export-test-h5ad \
    --timestamp "$TS" \
    --scheduler-min-lr 1e-5 \
    --m5c-bedgraph "${m5c_paths[@]}" \
    --hm5c-bedgraph "${hm5c_paths[@]}" \
    --atac-bw "${atac_paths[@]}" \
    --rna-coverage-bw "${rna_paths[@]}" \
    --output-csv "${OUT_ROOT}/${TS}_${label}_results.csv" \
    --output-json "${OUT_ROOT}/${TS}_${label}_results.json" \
    --best-checkpoint-path "${OUT_ROOT}/${TS}_${label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${OUT_ROOT}/${TS}_${label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log"
  local code=${PIPESTATUS[0]}
  echo "[$(date)] ${tag} exit=${code}" | tee -a "$log"
  return $code
}

pids=()
for i in "${!drops[@]}"; do
  drop="${drops[$i]}"
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  tag="$(printf 'd%s' "$drop" | tr -d '.')"
  run_one "$tag" "$drop" "$gpu" &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
echo "[$(date)] pilots done; failed=${failed}"
[[ $failed -eq 0 ]]
