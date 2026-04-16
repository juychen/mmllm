#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

region=PFC
current_time=$(date "+%Y-%m-%d-%H-%M-%S")

echo "[$region] Current time: $current_time"
python run_sample_size_experiments.py \
  --sample-sizes 2000 20000 100000 \
  --use-sequence --no-use-atac \
  --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_${region}.CG.m.bedGraph.gz \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_${region}.CG.h.bedGraph.gz \
  --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_MC_track.bw \
  --scheduler cosine \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --timestamp "$current_time" \
  --scheduler-min-lr 1e-5 \
  --output-csv output/${current_time}_${region}_mc_hmc_results.csv \
  --output-json output/${current_time}_${region}_mc_hmc_results.json \
  --prediction-signal-csv output/${current_time}_${region}_prediction_mchmc_{sample_size}.csv \
  --regression-plot-path output/${current_time}_${region}_regression_mchmc_{sample_size}.png

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
echo "[$region] Current time: $current_time"
python run_atac_query_sequence_context_experiments.py \
  --sample-sizes 2000 20000 100000 \
  --scheduler cosine \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_${region}.CG.m.bedGraph.gz \
  --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_MC_track.bw \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --scheduler-min-lr 1e-5 \
  --timestamp "$current_time" \
  --output-csv output/${current_time}_${region}_atac_5mc_results.csv \
  --output-json output/${current_time}_${region}_atac_5mc_results.json \
  --prediction-signal-csv output/${current_time}_${region}_prediction_5mc_{sample_size}.csv \
  --regression-plot-path output/${current_time}_${region}_regression_5mc_{sample_size}.png

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
echo "[$region] Current time: $current_time"
python run_atac_query_sequence_context_experiments.py \
  --sample-sizes 2000 20000 100000 \
  --scheduler cosine \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_${region}.CG.h.bedGraph.gz \
  --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_MC_track.bw \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --scheduler-min-lr 1e-5 \
  --timestamp "$current_time" \
  --output-csv output/${current_time}_${region}_atac_5h_results.csv \
  --output-json output/${current_time}_${region}_atac_5h_results.json \
  --prediction-signal-csv output/${current_time}_${region}_prediction_5h_{sample_size}.csv \
  --regression-plot-path output/${current_time}_${region}_regression_5h_{sample_size}.png