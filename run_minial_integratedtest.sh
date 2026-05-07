#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

region="${1:-ALL}"
condition="${2:-MC}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      echo "Usage: $0 [REGION|ALL] [CONDITION]"
      echo "Example: $0 AMY MC"
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

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
if [[ "$region" == "ALL" ]]; then
  output_dir="output/ALL_GROUPS"
else
  output_dir="output/${region}_${condition}"
fi

echo "[$region] Current time: $current_time"

common_args=(
  --sample-sizes 5000 20000 50000 200000 500000
  --augment-reverse-complement
  --mask-mode cpg_forward
  --scheduler cosine
  --num-epochs 100
  --batch-size 64
  --scheduler-patience 5
  --timestamp "$current_time"
  --scheduler-min-lr 1e-5
  --output-csv "${output_dir}/${current_time}_multi_integrated_results.csv"
  --output-json "${output_dir}/${current_time}_multi_integrated_results.json"
  --prediction-signal-csv "${output_dir}/${current_time}_multi_integrated_{sample_size}.csv"
  --regression-plot-path "${output_dir}/${current_time}_multi_integrated_{sample_size}.png"
)

if [[ "$region" == "ALL" ]]; then
  python run_multimodal_multitask_experiments.py \
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
      /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/PFC_MW_track.bw
else
  python run_multimodal_multitask_experiments.py \
    "${common_args[@]}" \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw"
fi