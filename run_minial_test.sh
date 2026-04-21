#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

region="${1:-AMY}"
condition="${2:-MC}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      echo "Usage: $0 [REGION] [CONDITION]"
      echo "Example: $0 AMY MC"
      exit 0
      ;;
  esac
fi

if [[ "$region" != "AMY" && "$region" != "HIP" && "$region" != "PFC" ]]; then
  echo "Unsupported region: $region"
  echo "Allowed values: AMY HIP PFC"
  exit 1
fi

if [[ "$condition" != "MC" && "$condition" != "MW" ]]; then
  echo "Unsupported condition: $condition"
  echo "Allowed values: MC MW"
  exit 1
fi

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
output_dir="output/${region}_${condition}"

echo "[$region] Current time: $current_time"
python run_multimodal_track_experiments.py \
  --sample-sizes 2000 10000 20000 50000 100000\
  --input-modality atac \
  --augment-reverse-complement \
  --context-modalities sequence \
  --target-modality 5hmc \
  --mask-mode cpg_both \
  --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz \
  --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw \
  --scheduler cosine \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --timestamp "$current_time" \
  --scheduler-min-lr 1e-5 \
  --output-csv ${output_dir}/${current_time}_atac_to_5hmc_results.csv \
  --output-json ${output_dir}/${current_time}_atac_to_5hmc_results.json \
  --prediction-signal-csv ${output_dir}/${current_time}_atac_to_5hmc_{sample_size}.csv \
  --regression-plot-path ${output_dir}/${current_time}_atac_to_5hmc_{sample_size}.png

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
echo "[$region] Current time: $current_time"
python run_multimodal_track_experiments.py \
  --sample-sizes 2000 10000 20000 50000 100000\
  --input-modality 5mc \
  --augment-reverse-complement \
  --context-modalities sequence \
  --target-modality 5hmc \
  --mask-mode cpg_both \
  --scheduler cosine \
  --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz \
  --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --scheduler-min-lr 1e-5 \
  --timestamp "$current_time" \
  --output-csv ${output_dir}/${current_time}_5mc_to_5hmc_results.csv \
  --output-json ${output_dir}/${current_time}_5mc_to_5hmc_results.json \
  --prediction-signal-csv ${output_dir}/${current_time}_5mc_to_5hmc_{sample_size}.csv \
  --regression-plot-path ${output_dir}/${current_time}_5mc_to_5hmc_{sample_size}.png

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
echo "[$region] Current time: $current_time"
python run_multimodal_track_experiments.py \
  --sample-sizes 2000 10000 20000 50000 100000\
  --input-modality atac \
  --augment-reverse-complement \
  --context-modalities sequence \
  --target-modality 5mc \
  --mask-mode cpg_both \
  --scheduler cosine \
  --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz \
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz \
  --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw \
  --num-epochs 100 \
  --batch-size 64 \
  --scheduler-patience 5 \
  --scheduler-min-lr 1e-5 \
  --timestamp "$current_time" \
  --output-csv ${output_dir}/${current_time}_atac_to_5mc_results.csv \
  --output-json ${output_dir}/${current_time}_atac_to_5mc_results.json \
  --prediction-signal-csv ${output_dir}/${current_time}_atac_to_5mc_{sample_size}.csv \
  --regression-plot-path ${output_dir}/${current_time}_atac_to_5mc_{sample_size}.png