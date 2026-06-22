#!/bin/bash
source /home/zhangyr/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /data1st2/zhangyr/code/mmllm/ || exit 1

region="${1:-AMY}"
condition="${2:-MC}"
fusion_type="${3:-cross_hyena}"

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

select_idle_gpu() {
  # 手动指定优先：
  # GPU_ID=2 bash run.sh AMY MC
  if [[ -n "${GPU_ID:-}" ]]; then
    echo "$GPU_ID"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Cannot auto-select GPU." >&2
    return 1
  fi

  # 最低要求空闲显存，单位 MiB
  # 你的任务接近 80G，建议至少要求 76000 MiB 以上
  # 用法：MIN_FREE_MB=76000 bash run.sh AMY MC
  local min_free_mb="${MIN_FREE_MB:-76000}"

  local selected_gpu
  selected_gpu=$(
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
      | awk -F',' -v min_free="$min_free_mb" '{
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          gsub(/ /, "", $3);
          if ($2 >= min_free) {
            print $1, $2, $3
          }
        }' \
      | sort -k2,2nr -k3,3n \
      | head -n 1 \
      | awk '{print $1}'
  )

  if [[ -z "$selected_gpu" ]]; then
    echo "ERROR: No GPU has at least ${min_free_mb} MiB free memory." >&2
    echo "Current GPU status:" >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv >&2
    return 1
  fi

  echo "$selected_gpu"
}


run_experiment() {
  selected_gpu="$(select_idle_gpu)" || exit 1
  export CUDA_VISIBLE_DEVICES="$selected_gpu"

  # 这个可以缓解部分显存碎片问题，但不能解决模型本身显存需求超过 80G 的问题
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo "Selected physical GPU: $selected_gpu"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi -i "$selected_gpu" --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  run_label="m5c_query_sequence_atac_crosshyena_modelb_${fusion_type}"
  
  if [[ "$region" == "ALL" ]]; then
    output_dir="output/ALL_GROUPS"
  else
    output_dir="/data1st2/zhangyr/data/mmllm/modelb/modelb_blocks2/hidden_dim128/${region}_${condition}"
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
    --hidden-dim 128
    --target-length 8192
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