#!/bin/bash
set -e

source /home/zhangyr/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /data1st2/zhangyr/code/mmllm/ || exit 1

fusion_type="${1:-cross_hyena}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      echo "Usage: $0 [cross_hyena|cross_attention]"
      echo "Example: $0"
      echo "Example: $0 cross_attention"
      exit 0
      ;;
  esac
fi

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

run_experiment() {
  local region="$1"
  local condition="$2"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  run_label="m5c_query_sequence_atac_crosshyena_modelb_${fusion_type}"

  output_dir="/data1st2/zhangyr/data/mmllm/modelb/modelb_blocks3/${region}_${condition}"
  mkdir -p "$output_dir"

  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "============================================================"
  echo "Running experiment: ${region}_${condition}"
  echo "Fusion type: $fusion_type"
  echo "Current time: $current_time"
  echo "Log file: $log_file"
  echo "============================================================"

  common_args=(
    --sample-sizes 500000
    --model-name model_b
    --model-b-blocks 3
    --model-b-fusion "$fusion_type"
    --augment-reverse-complement
    --mask-mode cpg_forward
    --scheduler cosine
    --num-epochs 100
    --batch-size 32
    --target-length 16384
    --scheduler-patience 15
    --timestamp "$current_time"
    --scheduler-min-lr 1e-5
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv"
    --output-json "${output_dir}/${current_time}_${run_label}_results.json"
    --prediction-signal-csv "${output_dir}/${current_time}_${run_label}_{sample_size}.csv"
    --regression-plot-path "${output_dir}/${current_time}_${run_label}_{sample_size}.png"
    --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt"
    --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt"
  )

  python run_m5c_query_sequence_atac_crosshyena_experiments.py \
    "${common_args[@]}" \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw" \
    2>&1 | tee "$log_file"
}

regions=("AMY" "HIP" "PFC")
conditions=("MC" "MW")

for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    run_experiment "$region" "$condition"
  done
done
