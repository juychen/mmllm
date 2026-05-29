#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

region="${1:-AMY}"
condition="${2:-MC}"
fusion_type="${3:-cross_attention}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      echo "Usage: $0 [REGION|ALL] [CONDITION]"
      echo "Usage: $0 [REGION|ALL] [CONDITION] [cross_hyena|cross_attention]"
      echo "Example: $0 AMY MC"
      echo "Example: $0 AMY MC cross_attention"
      echo "Example: $0 ALL"
      exit 0
      ;;
  esac
fi

if [[ "$region" != "AMY" && "$region" != "HIP" && "$region" != "PFC" && "$region" != "ALL" ]]; then
  echo "Unsupported region: $region"
  echo "Allowed values: AMY HIP PFC ALL"
  exit 1
fi

if [[ "$region" != "ALL" && "$condition" != "MC" && "$condition" != "MW" ]]; then
  echo "Unsupported condition: $condition"
  echo "Allowed values: MC MW"
  exit 1
fi

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

run_experiment() {
  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  run_label="m5c_query_sequence_atac_crosshyena_modelb_${fusion_type}"
  if [[ "$region" == "ALL" ]]; then
    output_dir="output/ALL_GROUPS"
  else
    output_dir="output/${region}_${condition}"
  fi
  mkdir -p "$output_dir"
  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$region/${condition}] Current time: $current_time"
  echo "[$region/${condition}] Log file: $log_file"

  common_args=(
    --sample-sizes 500000
    --model-name model_b
    --model-b-blocks 2
    --model-b-fusion "$fusion_type"
    --augment-reverse-complement
    --mask-mode cpg_forward
    --scheduler cosine
    --num-epochs 100
    --batch-size 64
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

  if [[ "$region" == "ALL" ]]; then
    python run_m5c_query_sequence_atac_crosshyena_experiments.py \
      "${common_args[@]}" \
      --use-all-input-groups \
      --m5c-bedgraph \
        /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MW_AMY.CG.m.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MC_HIP.CG.m.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MW_HIP.CG.m.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.m.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MW_PFC.CG.m.bedGraph.gz \
      --hm5c-bedgraph \
        /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MW_AMY.CG.h.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MC_HIP.CG.h.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MW_HIP.CG.h.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.h.bedGraph.gz \
        /data2st1/junyi/output/llm0401/processed_meth/MW_PFC.CG.h.bedGraph.gz \
      --atac-bw \
        /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw \
        /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MW_track.bw \
        /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/HIP_MC_track.bw \
        /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/HIP_MW_track.bw \
        /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/PFC_MC_track.bw \
        /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/PFC_MW_track.bw \
      2>&1 | tee "$log_file"
  else
    python run_m5c_query_sequence_atac_crosshyena_experiments.py \
      "${common_args[@]}" \
      --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
      --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
      --atac-bw "/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw" \
      2>&1 | tee "$log_file"
  fi
}

run_experiment