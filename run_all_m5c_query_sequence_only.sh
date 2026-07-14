#!/bin/bash
# Parallel runner for sequence-only (ATAC-optional) ablation on AMY/HIP/PFC x MC/MW.
# Each call runs ONE experiment (5mC + sequence (+ optional ATAC) -> 5hmC).
# Usage: ./run_all_m5c_query_sequence_only.sh [fusion_type]
#   fusion_type: cross_hyena (default) | cross_attention

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported fusion_type: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

# Toggle ATAC branch (set to 1 to include ATAC, 0 to omit it).
# For an ATAC-ablation sweep, run the script twice with USE_ATAC=0 and USE_ATAC=1.
USE_ATAC="${USE_ATAC:-0}"

regions=("AMY" "HIP" "PFC")
conditions=("MC" "MW")

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"
  local use_atac="$4"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  local dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  local atac_flag=""
  local atac_arg=()
  if [[ "$use_atac" == "1" ]]; then
    atac_flag="with_atac"
    atac_arg=(--use-atac --atac-bw "/data1st1/junyi/methdata/atac/${region}_${condition}_track.bw")
  else
    atac_flag="seq_only"
    atac_arg=()
  fi

  local run_label="m5c_query_sequence_only_modelb_${fusion_type}_${atac_flag}"
  local output_dir="output/${region}_${condition}/${bed_name}/${run_label}"
  mkdir -p "$output_dir"
  local log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] ${atac_flag} Starting... (BED: ${bed_name})" | tee -a "$log_file"

  python run_m5c_query_sequence_only_experiments.py \
    --sample-sizes all \
    --dmr-csv "$dmr_csv" \
    --model-b-blocks 2 \
    --model-b-fusion "$fusion_type" \
    --augment-reverse-complement \
    --hidden-dim 512 \
    --mask-mode cpg_forward \
    --scheduler cosine \
    --num-epochs 100 \
    --batch-size 4 \
    --target-length 16384 \
    --gradient-accumulation-steps 64 \
    --scheduler-patience 15 \
    --patience 15 \
    --amp \
    --gradient-checkpointing \
    --lazy \
    --timestamp "$current_time" \
    --scheduler-min-lr 1e-5 \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    "${atac_arg[@]}" \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
    --prediction-signal-csv "${output_dir}/${current_time}_${run_label}_{sample_size}.csv" \
    --regression-plot-path "${output_dir}/${current_time}_${run_label}_{sample_size}.png" \
    --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -eq 0 ]; then
    echo "[$(date)] [${region}_${condition}] ${atac_flag} Finished successfully!" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] ${atac_flag} FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

MAX_CONCURRENT="${MAX_CONCURRENT:-3}"

echo ""
echo "============================================"
echo "[$(date)] Submitting sequence-only experiments (fusion=${fusion_type}, use_atac=${USE_ATAC}, max ${MAX_CONCURRENT} concurrent)..."
echo "============================================"

failed=0
total=0
running=0

for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
      wait -n
      running=$((running - 1))
    fi

    run_experiment "$region" "$condition" "$fusion_type" "$USE_ATAC" &
    running=$((running + 1))
    total=$((total + 1))
  done
done

echo "[$(date)] Waiting for the last ${running} job(s) to complete..."
for job in $(jobs -p); do
  wait "$job" || { failed=$((failed + 1)); }
done

echo ""
echo "============================================"
if [ $failed -eq 0 ]; then
  echo "[$(date)] All ${total} experiments completed successfully!"
else
  echo "[$(date)] $failed out of ${total} experiments FAILED!"
fi
echo "============================================"
