#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/
# GET current time
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
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
condition=MC
region=AMY

echo "Current time: $current_time"
python run_multimodal_track_experiments.py \
  --sample-sizes 2000 \
  --input-modality atac \
  --context-modalities sequence \
  --target-modality 5hmc

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
python run_multimodal_track_experiments.py \
  --sample-sizes 2000 \
  --input-modality 5mc \
  --context-modalities sequence \
  --target-modality 5hmc

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
python run_multimodal_track_experiments.py \
  --sample-sizes 2000 \
  --input-modality atac \
  --context-modalities sequence \
  --target-modality 5mc