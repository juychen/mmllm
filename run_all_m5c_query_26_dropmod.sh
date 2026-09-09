#!/bin/bash
# Modality-dropout (mask-token) training for model_b + RNA: AMY/HIP/PFC x MC/MW.
# Wraps run_m5c_query_crosshyena_dropmod.py (--seq-drop-p / --atac-drop-p / --rna-drop-p).
#
# Usage:
#   ./run_all_m5c_query_crosshyena_dropmod.sh [fusion_type] [seq_drop_p] [atac_drop_p] [rna_drop_p]
# Example:
#   ./run_all_m5c_query_crosshyena_dropmod.sh cross_hyena 0.1 0.2 0.1
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"
seq_drop_p="${2:-0.0}"
atac_drop_p="${3:-0.2}"
rna_drop_p="${4:-0.2}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

# Validate dropout probabilities are in [0, 1]
for prob in "$seq_drop_p" "$atac_drop_p" "$rna_drop_p"; do
  if ! awk -v p="$prob" 'BEGIN{exit !(p >= 0 && p <= 1)}'; then
    echo "Dropout probability out of range [0,1]: $prob"
    exit 1
  fi
done

regions=("AMY")
conditions=("MC" "MW")

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")

  # Extract a short BED identifier from --dmr-csv
  local dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  run_label="m5c_query_crosshyena_modelb_${fusion_type}_dropmod_s${seq_drop_p}_a${atac_drop_p}_r${rna_drop_p}"
  output_dir="output/${region}_${condition}/${bed_name}"
  mkdir -p "$output_dir"
  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] Starting... (BED: ${bed_name}, drops: seq=${seq_drop_p} atac=${atac_drop_p} rna=${rna_drop_p})" | tee -a "$log_file"

  python run_m5c_query_crosshyena_dropmod.py \
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
    --seq-drop-p "$seq_drop_p" \
    --atac-drop-p "$atac_drop_p" \
    --rna-drop-p "$rna_drop_p" \
    --m5c-bedgraph "/data8/junyi/methdata/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data8/junyi/methdata/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data8/junyi/methdata/atac/${region}_${condition}_track.bw" \
    --rna-coverage-bw "/data8/junyi/methdata/bulk_rna/BULK_${region}/${region}_${condition}.bw" \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
    --prediction-signal-h5ad "${output_dir}/${current_time}_${run_label}_{sample_size}.h5ad" \
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
echo "[$(date)] Submitting modality-dropout experiments (max ${MAX_CONCURRENT} concurrent)..."
echo "  fusion=${fusion_type}, drops: seq=${seq_drop_p} atac=${atac_drop_p} rna=${rna_drop_p}"
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
