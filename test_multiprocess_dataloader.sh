#!/bin/bash
# Quick test: verify multi-process DataLoader works with lazy loading.
# Uses minimal params — ~2k target length, 100 samples, 1 epoch.
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
output_dir="output/test_multiprocess"
mkdir -p "$output_dir"
log_file="${output_dir}/${current_time}_test.log"

echo "[$(date)] Starting multi-process DataLoader test..." | tee "$log_file"

python run_m5c_query_sequence_atac_crosshyena_experiments.py \
  --sample-sizes 2000 \
  --model-name model_b \
  --model-b-blocks 1 \
  --model-b-fusion cross_hyena \
  --target-length 1024 \
  --hidden-dim 16 \
  --num-epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --scheduler none \
  --patience 0 \
  --amp \
  --gradient-checkpointing \
  --lazy \
  --timestamp "$current_time" \
  --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz \
  --atac-bw /data1st1/junyi/methdata/atac/AMY_MC_track.bw \
  --output-csv "${output_dir}/${current_time}_test_results.csv" \
  --output-json "${output_dir}/${current_time}_test_results.json" \
  --prediction-signal-h5ad "${output_dir}/${current_time}_test_{sample_size}.h5ad" \
  --regression-plot-path "${output_dir}/${current_time}_test_{sample_size}.png" \
  --best-checkpoint-path "${output_dir}/${current_time}_test_best_{sample_size}.pt" \
  --last-checkpoint-path "${output_dir}/${current_time}_test_last_{sample_size}.pt" \
  2>&1 | tee -a "$log_file"

exit_code=${PIPESTATUS[0]}
echo "[$(date)] Exit code: $exit_code" | tee -a "$log_file"
if [ $exit_code -eq 0 ]; then
  echo "[$(date)] SUCCESS: multi-process DataLoader works!" | tee -a "$log_file"
else
  echo "[$(date)] FAILED (exit code $exit_code). Check log: $log_file" | tee -a "$log_file"
fi
