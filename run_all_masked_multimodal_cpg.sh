#!/bin/bash
# Run masked MULTIMODAL pretraining for AMY/HIP/PFC x MC/MW — parallel submission.
#
# Calls run_masked_multimodal_pretraining.py:
#   * inputs: 5mC / 5hmC / ATAC / RNA + DNA sequence
#   * output heads: 5mc + 5hmc only (ATAC/RNA are input-only)
#   * position mask: shared across all signal tracks (learned [POS-MASK] tokens)
#   * modality mask: per-track learned [MODALITY-MASK] tokens, 方案①
#     (every step, every sample independently re-draws each track's presence)
#   * loss only at position-masked positions -> anti-leak by construction
#
# Usage:
#   ./run_all_masked_multimodal_cpg.sh [fusion_type] [seq_drop_p] [atac_drop_p] [rna_drop_p]
# Example:
#   ./run_all_masked_multimodal_cpg.sh cross_hyena 0.0 0.2 0.2
#
# Optional env vars:
#   TEST_BED=/path/to/holdout.bed   held-out test regions (chromosome or >=10Mb block BED)
#   MASK_FRACTION=0.15              shared CpG position-mask fraction
#
# Tip: smoke-test first with a small run before the full submission, e.g. add
#   --sample-sizes 2000 --num-epochs 2
# to the python call below.
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"
seq_drop_p="${2:-0.0}"
atac_drop_p="${3:-0.0}"
rna_drop_p="${4:-0.0}"
mask_fraction="${MASK_FRACTION:-0.15}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported fusion: $fusion_type"
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

  # Extract a short BED identifier from --dmr-csv
  local dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  local run_label="masked_multimodal_pretrain_s${seq_drop_p}_a${atac_drop_p}_r${rna_drop_p}"
  local output_dir="output/${region}_${condition}/${bed_name}"
  mkdir -p "$output_dir"
  local log_file="${output_dir}/${current_time}_${run_label}.log"

  # Optional held-out test BED (chromosome-level holdout recommended)
  local test_args=()
  if [[ -n "${TEST_BED:-}" ]]; then
    test_args=(--test-bed "$TEST_BED")
  fi

  echo "[$(date)] [${region}_${condition}] Starting masked multimodal pretraining (BED: ${bed_name})..." | tee -a "$log_file"
  echo "[$(date)] [${region}_${condition}] drops: seq=${seq_drop_p} atac=${atac_drop_p} rna=${rna_drop_p} | mask_fraction=${mask_fraction} | test_bed=${TEST_BED:-none}" | tee -a "$log_file"

  python run_masked_multimodal_pretraining.py \
    --sample-sizes all \
    --dmr-csv "$dmr_csv" \
    --target-length 16384 \
    --num-blocks 2 \
    --fusion-type "$fusion_type" \
    --reconstruct-tracks 5mc 5hmc \
    --mask-fraction "$mask_fraction" \
    --drop-p-seq "$seq_drop_p" \
    --drop-p-atac "$atac_drop_p" \
    --drop-p-rna "$rna_drop_p" \
    --augment-reverse-complement \
    --mask-mode cpg_forward \
    --scheduler cosine \
    --num-epochs 100 \
    --batch-size 4 \
    --gradient-accumulation-steps 64 \
    --scheduler-patience 15 \
    --patience 15 \
    --amp \
    --gradient-checkpointing \
    --timestamp "$current_time" \
    --scheduler-min-lr 1e-5 \
    "${test_args[@]}" \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data1st1/junyi/methdata/atac/${region}_${condition}_track.bw" \
    --rna-coverage-bw "/data2st1/junyi/output/sn0615/BULK_${region}/${region}_${condition}.bw" \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
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
echo "[$(date)] Submitting masked multimodal pretraining (max ${MAX_CONCURRENT} concurrent)..."
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
