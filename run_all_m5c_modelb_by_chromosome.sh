#!/bin/bash
# Run m5c model B experiments separated by chromosome.
#
# Usage:
#   bash run_all_m5c_modelb_by_chromosome.sh [fusion_type] [chromosomes...]
#
# Examples:
#   # Run all chromosomes
#   bash run_all_m5c_modelb_by_chromosome.sh cross_hyena all
#
#   # Run specific chromosomes only
#   bash run_all_m5c_modelb_by_chromosome.sh cross_hyena chr1 chr2 chr3
#
#   # Run autosomes only (no sex chromosomes)
#   bash run_all_m5c_modelb_by_chromosome.sh cross_hyena auto
#
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"
shift 2>/dev/null || true

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

ALL_CHROMOSOMES=(
  chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
  chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19
  chrX chrY
)
AUTO_CHROMOSOMES=(
  chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
  chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19
)

# Determine which chromosomes to run
if [ $# -eq 0 ]; then
  echo "Usage: $0 [fusion_type] [chromosomes...]"
  echo "  e.g. $0 cross_hyena chr1 chr2"
  echo "  e.g. $0 cross_hyena all"
  echo "  e.g. $0 cross_hyena auto"
  exit 1
fi

chromosomes=("$@")
if [ "${chromosomes[0]}" = "all" ]; then
  chromosomes=("${ALL_CHROMOSOMES[@]}")
elif [ "${chromosomes[0]}" = "auto" ]; then
  chromosomes=("${AUTO_CHROMOSOMES[@]}")
fi

# Config
dmr_csv="/data2st1/junyi/generegion_vM23/cCREs_mm10encode.bed"
bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"
MAX_CONCURRENT=4

run_experiment() {
  local region="$1"
  local condition="$2"
  local chromosome="$3"
  local fusion_type="$4"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")

  run_label="m5c_query_sequence_atac_crosshyena_modelb_${fusion_type}_${chromosome}"
  output_dir="output/${region}_${condition}/${bed_name}/${chromosome}"
  mkdir -p "$output_dir"
  log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition} ${chromosome}] Starting..." | tee -a "$log_file"

  python run_m5c_query_sequence_atac_crosshyena_experiments.py \
    --sample-sizes all \
    --dmr-csv "$dmr_csv" \
    --chromosome "$chromosome" \
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
    echo "[$(date)] [${region}_${condition} ${chromosome}] Finished successfully!" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition} ${chromosome}] FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

echo ""
echo "============================================"
echo "[$(date)] Submitting per-chromosome experiments (max ${MAX_CONCURRENT} concurrent)..."
echo "============================================"
printf "Regions:     %s\n" "${regions[*]}"
printf "Conditions:  %s\n" "${conditions[*]}"
printf "Chromosomes: %s\n" "${chromosomes[*]}"
echo "Total jobs:  ${#regions[@]} x ${#conditions[@]} x ${#chromosomes[@]} = $(( ${#regions[@]} * ${#conditions[@]} * ${#chromosomes[@]} ))"
echo ""

failed=0
total=0
running=0

for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    for chromosome in "${chromosomes[@]}"; do
      # Wait if we already have MAX_CONCURRENT jobs running
      if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
      fi

      run_experiment "$region" "$condition" "$chromosome" "$fusion_type" &
      running=$((running + 1))
      total=$((total + 1))
    done
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
