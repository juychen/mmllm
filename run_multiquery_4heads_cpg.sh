#!/bin/bash
# Multi-query masked pretraining with FOUR output heads:
#   5mC, 5hmC, ATAC and RNA each get their own query stream + output head,
#   and each one sees DNA sequence + the OTHER three tracks as its context
#   (context_mode = all-others).  Nothing is "context-only" any more.
#
# Model: MultiQueryMaskedPretrainingModel, hidden 64 / 2 blocks / cross_hyena,
#        16 kb windows, 15% shared CpG mask, reverse-complement augmentation.
#
# Usage:
#   ./run_multiquery_4heads_cpg.sh                 # AMY_MC, one GPU
#   REGIONS="AMY HIP PFC" CONDITIONS="MC MW" GPUS="5 7" MAX_CONCURRENT=2 \
#     ./run_multiquery_4heads_cpg.sh               # the full 6-group sweep
#   GPUS=7 ./run_multiquery_4heads_cpg.sh          # pick the card explicitly
#   N_SMOKE=1 OUT_ROOT=/tmp/mq4_smoke ./run_multiquery_4heads_cpg.sh
#
# Output: output/<REGION>_<CONDITION>/cCRE_cpg/<timestamp>_multiquery_4heads_*.{log,pt,csv,json}
#
# NOTE on the loss: run_multiquery_pretraining.py averages the per-track MSE
# without weights, so the 0-100 methylation targets dominate the shared-block
# gradients; the ATAC/RNA heads still train (their own projections + heads get
# gradients) but contribute little to the trunk.  Check per-track R^2 in the
# result JSON; if the ATAC/RNA heads underfit, per-track loss weights would
# need a code change.
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

export PYTHONUNBUFFERED=1

context_mode="${CONTEXT_MODE:-all-others}"
mask_fraction="${MASK_FRACTION:-0.15}"
drop_5mc="${DROP_5MC:-0.0}"
drop_5hmc="${DROP_5HMC:-0.0}"
drop_atac="${DROP_ATAC:-0.0}"
drop_rna="${DROP_RNA:-0.0}"
drop_seq="${DROP_SEQ:-0.0}"
run_tag="${RUN_TAG:-}"
batch_size="${BATCH_SIZE:-16}"          # 4 streams need ~2x the memory of 2 heads
grad_accum="${GRAD_ACCUM:-16}"          # keep batch_size * grad_accum = 256
num_workers="${NUM_WORKERS:-8}"
num_epochs="${NUM_EPOCHS:-100}"
patience="${PATIENCE:-15}"
regions=(${REGIONS:-AMY})
conditions=(${CONDITIONS:-MC})
gpus=(${GPUS:-5})
max_concurrent="${MAX_CONCURRENT:-${#gpus[@]}}"
out_root="${OUT_ROOT:-output}"
sample_sizes="${SAMPLE_SIZES:-all}"

if [[ -n "${N_SMOKE:-}" ]]; then
  sample_sizes=64
  num_epochs=1
  batch_size=16
  grad_accum=1
  num_workers=0
  echo "[smoke] 64 windows, 1 epoch, batch 16"
fi

run_experiment() {
  local region="$1"
  local condition="$2"
  local gpu="$3"

  local ts
  ts=$(date "+%Y-%m-%d-%H-%M-%S")
  local dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  local run_label="multiquery_4heads_ctx${context_mode}${run_tag:+_${run_tag}}"
  local output_dir="${out_root}/${region}_${condition}/${bed_name}"
  mkdir -p "$output_dir"
  local log_file="${output_dir}/${ts}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] GPU${gpu} | 4 heads: 5mc+5hmc+atac+rna | context=${context_mode}" | tee -a "$log_file"
  echo "[$(date)] [${region}_${condition}] bs=${batch_size} accum=${grad_accum} workers=${num_workers} epochs=${num_epochs} mask=${mask_fraction} | drop: 5mc=${drop_5mc} 5hmc=${drop_5hmc} atac=${drop_atac} rna=${drop_rna} seq=${drop_seq}" | tee -a "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" python run_multiquery_pretraining.py \
    --sample-sizes "$sample_sizes" \
    --dmr-csv "$dmr_csv" \
    --target-length 16384 \
    --num-blocks 2 \
    --hidden-dim 64 \
    --fusion-type cross_hyena \
    --query-tracks 5mc 5hmc atac rna \
    --context-mode "$context_mode" \
    --mask-fraction "$mask_fraction" \
    --drop-p-5mc "$drop_5mc" --drop-p-5hmc "$drop_5hmc" --drop-p-atac "$drop_atac" --drop-p-rna "$drop_rna" --drop-p-seq "$drop_seq" \
    --augment-reverse-complement \
    --mask-mode cpg_forward \
    --scheduler cosine \
    --num-epochs "$num_epochs" \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$grad_accum" \
    --num-workers "$num_workers" \
    --scheduler-patience "$patience" \
    --patience "$patience" \
    --amp \
    --timestamp "$ts" \
    --scheduler-min-lr 1e-5 \
    --m5c-bedgraph  "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data1st1/junyi/methdata/atac/${region}_${condition}_track.bw" \
    --rna-coverage-bw "/data2st1/junyi/output/sn0615/BULK_${region}/${region}_${condition}.bw" \
    --output-csv "${output_dir}/${ts}_${run_label}_results.csv" \
    --output-json "${output_dir}/${ts}_${run_label}_results.json" \
    --best-checkpoint-path "${output_dir}/${ts}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${output_dir}/${ts}_${run_label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local code=${PIPESTATUS[0]}
  if [[ $code -eq 0 ]]; then
    echo "[$(date)] [${region}_${condition}] finished" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] FAILED (exit $code)" | tee -a "$log_file"
  fi
  return $code
}

echo ""
echo "============================================"
echo "[$(date)] 4-head multiquery pretraining"
echo "  queries : 5mc 5hmc atac rna (each = stream + head + loss)"
echo "  context : DNA sequence + the other three tracks (${context_mode})"
echo "  groups  : ${regions[*]} x ${conditions[*]}"
echo "  GPUs    : ${gpus[*]} (max ${max_concurrent} concurrent)"
echo "  output  : ${out_root}/<REGION>_<CONDITION>/cCRE_cpg/"
echo "============================================"

failed=0
running=0
gpu_i=0
for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    if (( running >= max_concurrent )); then
      wait -n 2>/dev/null || true
      running=$((running - 1))
    fi
    gpu="${gpus[$((gpu_i % ${#gpus[@]}))]}"
    gpu_i=$((gpu_i + 1))
    run_experiment "$region" "$condition" "$gpu" &
    running=$((running + 1))
  done
done

echo "[$(date)] waiting for the last ${running} job(s) ..."
for job in $(jobs -p); do
  wait "$job" || failed=$((failed + 1))
done

echo ""
echo "============================================"
if [[ $failed -eq 0 ]]; then
  echo "[$(date)] all ${#regions[@]}x${#conditions[@]} group(s) finished"
else
  echo "[$(date)] ${failed} group(s) FAILED — check the .log next to each run"
fi
echo "============================================"
[[ $failed -eq 0 ]]
