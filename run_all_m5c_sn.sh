#!/bin/bash
# Run 5mC + sequence + phastCons -> 5hmC experiments on AMY/HIP/PFC x MC/MW.
#
# Rationale: replaces ATAC-seq with 60-way phastCons conservation scores as the
# second context modality. Since phastCons is a static mm10 bigWig, the input is
# identical for all regions/conditions — this isolates the contribution of
# sequence conservation from the cell-type-specific ATAC signal.
#
# Usage: ./run_all_m5c_phascon.sh [fusion_type]
#   fusion_type: cross_hyena (default) | cross_attention
#   MAX_CONCURRENT: 3 by default

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported fusion_type: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

# ---- config ----
PHASTCONS_BW="/data2st1/junyi/output/sn0615/BULK_AMY/AMY_MC.bw"

if [[ ! -f "$PHASTCONS_BW" ]]; then
  echo "ERROR: phastCons bigWig not found: $PHASTCONS_BW"
  exit 1
fi

regions=("AMY")
conditions=("MC")

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"

  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  local dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  local run_label="m5c_query_sequence_phascon_modelb_${fusion_type}"
  local output_dir="output/${region}_${condition}/${bed_name}/sn"
  mkdir -p "$output_dir"
  local log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] phastCons Starting... (BED: ${bed_name})" | tee -a "$log_file"

  python run_m5c_query_sequence_only_experiments.py \
    --sample-sizes all \
    --dmr-csv "$dmr_csv" \
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
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --use-atac \
    --atac-bw "$PHASTCONS_BW" \
    --atac-scaling minmax \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
    --prediction-signal-csv "${output_dir}/${current_time}_${run_label}_{sample_size}.csv" \
    --regression-plot-path "${output_dir}/${current_time}_${run_label}_{sample_size}.png" \
    --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -eq 0 ]; then
    echo "[$(date)] [${region}_${condition}] phastCons Finished successfully!" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] phastCons FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

MAX_CONCURRENT="${MAX_CONCURRENT:-3}"

echo ""
echo "============================================"
echo "[$(date)] Submitting phastCons experiments (fusion=${fusion_type}, max ${MAX_CONCURRENT} concurrent)..."
echo "[$(date)] phastCons bigWig: $PHASTCONS_BW"
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
  echo "[$(date)] All ${total} experiments completed successfully!"
else
  echo "[$(date)] $failed out of ${total} experiments FAILED!"
fi
echo "============================================"
