#!/bin/bash
# Run the new ablation preset `m5c_atac_attn` (query=m5c, context=[atac], cross-attention fusion)
# across AMY/HIP/PFC x MC/MW tissue / condition combinations. Parallel submission.
#
# Usage:
#   bash run_atac_ablation_m5c_atac_attn.sh              # default: 4 concurrent
#   bash run_atac_ablation_m5c_atac_attn.sh 2            # 2 concurrent

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

regions=("AMY" "HIP" "PFC")
conditions=("MC" "MW")

run_experiment() {
  local region="$1"
  local condition="$2"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")

  local dmr_csv="/data2st1/junyi/generegion_vM23/cCREs_mm10encode.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  run_label="atac_ablation_m5c_atac_attn"
  output_dir="output/${region}_${condition}/${bed_name}/m5c_atac_attn"
  mkdir -p "$output_dir"
  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] Starting... (BED: ${bed_name})" | tee -a "$log_file"

  python run_atac_ablation.py \
    --ablation m5c_atac_attn \
    --sample-sizes all \
    --dmr-csv "$dmr_csv" \
    --genome-fasta "/data2st1/junyi/ref/GRCm38.p6.genome.fa" \
    --hidden-dim 64 \
    --model-b-blocks 2 \
    --model-b-fusion cross_attention \
    --mask-mode cpg_forward \
    --atac-scaling minmax \
    --augment-reverse-complement \
    --scheduler cosine \
    --num-epochs 100 \
    --batch-size 2 \
    --target-length 16384 \
    --gradient-accumulation-steps 64 \
    --scheduler-patience 15 \
    --scheduler-min-lr 1e-5 \
    --amp \
    --gradient-checkpointing \
    --lazy \
    --timestamp "$current_time" \
    --output-dir "output/${region}_${condition}/${bed_name}" \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data1st1/junyi/methdata/atac/${region}_${condition}_track.bw" \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -eq 0 ]; then
    echo "[$(date)] [${region}_${condition}] Finished successfully!" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

MAX_CONCURRENT="${1:-4}"
echo "Running m5c_atac_attn ablation on ${#regions[@]} regions x ${#conditions[@]} conditions = $((${#regions[@]} * ${#conditions[@]})) jobs, max $MAX_CONCURRENT concurrent."

export -f run_experiment

parallel_count=0
for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    run_experiment "$region" "$condition" &
    parallel_count=$((parallel_count + 1))
    if [[ $parallel_count -ge $MAX_CONCURRENT ]]; then
      wait -n
      parallel_count=$((parallel_count - 1))
    fi
  done
done

wait
echo "[$(date)] All m5c_atac_attn ablation runs complete."
