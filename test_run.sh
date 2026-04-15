#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/
# GET current time
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
echo "Current time: $current_time"
# python run_sample_size_experiments.py \
#   --sample-sizes 50000\
#   --use-sequence --use-atac \
#   --scheduler cosine \
#   --num-epochs 100 \
#   --batch-size 64 \
#   --scheduler-patience 5 \
#   --scheduler-min-lr 1e-5 \
#   --output-csv output/$current_time\_sample_size_results.csv \
#   --output-json output/$current_time\_sample_size_results.json

# python run_sample_size_experiments.py \
#   --sample-sizes 50000\
#   --use-sequence --no-use-atac \
#   --scheduler cosine \
#   --num-epochs 100 \
#   --batch-size 64 \
#   --scheduler-patience 5 \
#   --scheduler-min-lr 1e-5 \
#   --output-csv output/$current_time\_sample_size_results.csv \
#   --output-json output/$current_time\_sample_size_results.json
# python run_sample_size_experiments.py \
#   --sample-sizes 50000\
#   --no-use-sequence --use-atac \
#   --scheduler cosine \
#   --num-epochs 100 \
#   --batch-size 64 \
#   --scheduler-patience 5 \
#   --scheduler-min-lr 1e-5 \
#   --output-csv output/$current_time\_sample_size_results.csv \
#   --output-json output/$current_time\_sample_size_results.json
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
python run_atac_query_sequence_context_experiments.py \
  --sample-sizes 2000 20000 100000\
  --scheduler cosine \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --scheduler-min-lr 1e-5 \
  --timestamp "$current_time" \
  --output-csv output/$current_time\_sample_atac_results.csv \
  --output-json output/$current_time\_sample_atac_results.json \
  --prediction-signal-csv output/$current_time\_prediction_signals_{sample_size}.csv \
  --regression-plot-path output/$current_time\_regression_plot_{sample_size}.png
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
python run_atac_query_sequence_context_experiments.py \
  --sample-sizes 2000 20000 100000\
  --scheduler cosine \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --scheduler-min-lr 1e-5 \
  --timestamp "$current_time" \
  --output-csv output/$current_time\_sample_atac_results.csv \
  --output-json output/$current_time\_sample_atac_results.json \
  --prediction-signal-csv output/$current_time\_prediction_signals_{sample_size}.csv \
  --regression-plot-path output/$current_time\_regression_plot_{sample_size}.png \