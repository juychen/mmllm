#!/bin/bash
# Run all AMY/HIP/PFC x MC/MW experiments for m5c model B
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

regions=("AMY" "HIP" "PFC")
conditions=("MC" "MW")
fusion_type="${1:-cross_hyena}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported model_b fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    current_time=$(date "+%Y-%m-%d-%H-%M-%S")
    run_label="m5c_query_sequence_atac_crosshyena_modelb_${fusion_type}"
    output_dir="output/${region}_${condition}"
    mkdir -p "$output_dir"
    log_file="${output_dir}/${current_time}_${run_label}.log"

    echo ""
    echo "============================================"
    echo "[$(date)] Starting: ${region} ${condition}"
    echo "============================================"

    python run_m5c_query_sequence_atac_crosshyena_experiments.py \
      --sample-sizes 2000 \
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
      2>&1 | tee "$log_file"

    echo "[$(date)] Finished: ${region} ${condition}"
  done
done

echo ""
echo "============================================"
echo "[$(date)] All experiments completed!"
echo "============================================"
