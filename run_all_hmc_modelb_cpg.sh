#!/bin/bash
# Run all AMY/HIP/PFC x MC/MW experiments for m5c model B — parallel submission
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

regions=("AMY")
conditions=("MW")

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")

  # Extract a short BED identifier from --dmr-csv
  local dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  # This run use hm5c bedgraph for methylation and m5c bedgraph for hydroxymethylation
  # Us 5mc as the target and hm5c as the input for the model
  run_label="hm5c_query_sequence_atac_crosshyena_modelb_${fusion_type}"
  output_dir="/data3/junyi/mmllm/output/${region}_${condition}/${bed_name}"
  mkdir -p "$output_dir"
  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] Starting... (BED: ${bed_name})" | tee -a "$log_file"
  echo "Output directory: $output_dir" | tee -a "$log_file"

  python run_m5c_query_sequence_atac_crosshyena_experiments.py \
    --sample-sizes all \
    --dmr-csv "$dmr_csv" \
    --model-name model_b \
    --model-b-blocks 2 \
    --model-b-fusion "$fusion_type" \
    --augment-reverse-complement \
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
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --atac-bw "/data1st1/junyi/methdata/atac/${region}_${condition}_track.bw" \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
    --prediction-signal-csv "${output_dir}/${current_time}_${run_label}_{sample_size}.csv" \
    --regression-plot-path "${output_dir}/${current_time}_${run_label}_{sample_size}.png" \
    --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -eq 0 ]; then
    echo "[$(date)] [${region}_${condition}] Finished successfully!" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

MAX_CONCURRENT=4

echo ""
echo "============================================"
echo "[$(date)] Submitting experiments (max ${MAX_CONCURRENT} concurrent)..."
echo "============================================"

failed=0
total=0
running=0

for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    # Wait if we already have MAX_CONCURRENT jobs running
    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
      wait -n
      running=$((running - 1))
    fi

    run_experiment "$region" "$condition" "$fusion_type" &
    running=$((running + 1))
    total=$((total + 1))
  done
done

# Wait for remaining background jobs and track exit codes
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
