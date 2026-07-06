#!/bin/bash
# Train m5c model B on chr1 — 16kb windows, 5% overlap
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

fusion_type="${1:-cross_hyena}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

regions=("AMY")
conditions=("MC")

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")

  local dmr_csv="/data1st1/junyi/output/mmllm/whole_genome_16kb_nonoverlap_beds/chr1_16kb_nonoverlap.bed"
  local bed_name="chr1_16kb_5p_overlap"

  run_label="m5c_chr1_modelb_${fusion_type}"
  output_dir="output/${region}_${condition}/${bed_name}"
  mkdir -p "$output_dir"
  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] Starting... (BED: ${bed_name})" | tee -a "$log_file"

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
    --batch-size 8 \
    --target-length 16384 \
    --gradient-accumulation-steps 64 \
    --scheduler-patience 15 \
    --amp \
    --gradient-checkpointing \
    --lazy \
    --timestamp "$current_time" \
    --scheduler-min-lr 1e-5 \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
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

MAX_CONCURRENT=2

echo ""
echo "============================================"
echo "[$(date)] chr1 16kb/5% overlap training — submitting ${#regions[@]} regions x ${#conditions[@]} conditions"
echo "Fusion: $fusion_type | Max concurrent: $MAX_CONCURRENT"
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

    run_experiment "$region" "$condition" "$fusion_type" &
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
  echo "[$(date)] All ${total} chr1 experiments completed successfully!"
else
  echo "[$(date)] $failed out of ${total} chr1 experiments FAILED!"
fi
echo "============================================"
