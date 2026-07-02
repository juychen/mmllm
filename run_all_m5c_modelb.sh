#!/bin/bash
# Run all AMY/HIP/PFC x MC/MW experiments for m5c model B — parallel submission
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

fusion_type="${1:-cross_hyena}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

regions=("HIP" "PFC")
conditions=("MC" "MW")

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  run_label="m5c_query_sequence_atac_crosshyena_modelb_${fusion_type}"
  output_dir="output/${region}_${condition}"
  mkdir -p "$output_dir"
  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] Starting..." | tee -a "$log_file"

  python run_m5c_query_sequence_atac_crosshyena_experiments.py \
    --sample-sizes 100000 \
    --dmr-csv /data2st1/junyi/generegion_vM23/cCRE_cpg.bed \
    --model-name model_b \
    --model-b-blocks 2 \
    --model-b-fusion "$fusion_type" \
    --augment-reverse-complement \
    --mask-mode cpg_forward \
    --scheduler cosine \
    --num-epochs 100 \
    --batch-size 2 \
    --target-length 16384 \
    --gradient-accumulation-steps 64 \
    --scheduler-patience 15 \
    --amp \
    --gradient-checkpointing \
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

echo ""
echo "============================================"
echo "[$(date)] Submitting all experiments in parallel..."
echo "============================================"

pids=()
for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    run_experiment "$region" "$condition" "$fusion_type" &
    pids+=($!)
  done
done

echo "[$(date)] Submitted ${#pids[@]} jobs. Waiting for all to complete..."
echo ""

# Wait for all background jobs and track exit codes
failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { failed=$((failed + 1)); }
done

echo ""
echo "============================================"
if [ $failed -eq 0 ]; then
  echo "[$(date)] All ${#pids[@]} experiments completed successfully!"
else
  echo "[$(date)] $failed out of ${#pids[@]} experiments FAILED!"
fi
echo "============================================"
