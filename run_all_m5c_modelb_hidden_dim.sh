#!/bin/bash
# Run hidden-dim ablation on AMY_MC for m5c model B — parallel submission
# Sweeps hidden_dim ∈ {128, 256} with blocks=2, blocks fixed.
# Usage: ./run_all_m5c_modelb_hidden_dim.sh [fusion_type]
#   fusion_type: cross_hyena (default) | cross_attention

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

# === Ablation grid ===
hidden_dims=(128 256)
num_blocks=2            # fixed — sweep hidden dim only

regions=("AMY")
conditions=("MC")

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"
  local hidden_dim="$4"
  local blocks="$5"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")

  # Extract a short BED identifier from --dmr-csv
  local dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  # Include hidden-dim in label and a dedicated subdir so runs don't collide
  local run_label="m5c_query_sequence_atac_crosshyena_modelb_${fusion_type}_hd${hidden_dim}_blk${blocks}"
  local output_dir="output/${region}_${condition}/${bed_name}/hd${hidden_dim}_blk${blocks}"
  mkdir -p "$output_dir"
  local log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] hd=${hidden_dim} blk=${blocks} Starting... (BED: ${bed_name})" | tee -a "$log_file"

  python run_m5c_query_sequence_atac_crosshyena_experiments.py \
    --sample-sizes all \
    --dmr-csv "$dmr_csv" \
    --model-name model_b \
    --model-b-blocks "$blocks" \
    --model-b-fusion "$fusion_type" \
    --augment-reverse-complement \
    --hidden-dim "$hidden_dim" \
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
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data1st1/junyi/methdata/atac/${region}_${condition}_track.bw" \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
    --prediction-signal-h5ad "${output_dir}/${current_time}_${run_label}_{sample_size}.h5ad" \
    --regression-plot-path "${output_dir}/${current_time}_${run_label}_{sample_size}.png" \
    --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -eq 0 ]; then
    echo "[$(date)] [${region}_${condition}] hd=${hidden_dim} blk=${blocks} Finished successfully!" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] hd=${hidden_dim} blk=${blocks} FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

MAX_CONCURRENT=2   # 2 hidden_dim × 1 region × 1 condition = 2 jobs; keep ≤2 to avoid OOM

echo ""
echo "============================================"
echo "[$(date)] Hidden-dim ablation: ${hidden_dims[*]} | blocks=${num_blocks} | fusion=${fusion_type}"
echo "[$(date)] Regions: ${regions[*]} | Conditions: ${conditions[*]}"
echo "[$(date)] Submitting experiments (max ${MAX_CONCURRENT} concurrent)..."
echo "============================================"

failed=0
total=0
running=0

for hd in "${hidden_dims[@]}"; do
  for region in "${regions[@]}"; do
    for condition in "${conditions[@]}"; do
      if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
      fi

      run_experiment "$region" "$condition" "$fusion_type" "$hd" "$num_blocks" &
      running=$((running + 1))
      total=$((total + 1))
    done
  done
done

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