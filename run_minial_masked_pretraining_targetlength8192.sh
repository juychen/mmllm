#!/bin/bash
source /home/zhangyr/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /data1st2/zhangyr/code/mmllm || exit 1

region="${1:-AMY}"
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

run_experiment() {
  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  if [[ "$region" == "ALL" ]]; then
    output_dir="/data1st2/zhangyr/data/mmllm/pretraining/target_length_8192"
  else
    output_dir="/data1st2/zhangyr/data/mmllm/pretraining/target_length_8192/${region}_${condition}"
  fi

  echo "[$region/${condition}] Current time: $current_time"

  common_args=(
    --sample-sizes 500000
    --reconstruct-tracks 5mc 5hmc
    --scheduler cosine
    --num-epochs 100
    --batch-size 64
    --target-length 8192
    --scheduler-patience 10
    --timestamp "$current_time"
    --scheduler-min-lr 1e-5
    --output-csv "${output_dir}/${current_time}_masked_pretraining_results.csv"
    --output-json "${output_dir}/${current_time}_masked_pretraining_results.json"
    --best-checkpoint-path "${output_dir}/${current_time}_masked_pretraining_best_{sample_size}.pt"
    --last-checkpoint-path "${output_dir}/${current_time}_masked_pretraining_last_{sample_size}.pt"
  )

  if [[ "$region" == "ALL" ]]; then
    python run_masked_track_pretraining.py \
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
    python run_masked_track_pretraining.py \
      "${common_args[@]}" \
      --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
      --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
      --atac-bw "/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw"
  fi
}

run_experiment
