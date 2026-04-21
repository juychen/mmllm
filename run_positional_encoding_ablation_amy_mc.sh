#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

region="AMY"
condition="MC"
sample_size=20000
seed="${1:-7}"
output_dir="output/${region}_${condition}"

echo "Running reverse-complement augmentation ablation for ${region}_${condition} with sample_size=${sample_size} seed=${seed}"

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
# echo "[without RC augmentation] timestamp=${current_time}"
# python run_multimodal_track_experiments.py \
#   --sample-sizes 5000 20000 \
#   --input-modality atac \
#   --context-modalities sequence \
#   --target-modality 5hmc \
#   --mask-mode cpg_both \
#   --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz \
#   --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz \
#   --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw \
#   --scheduler cosine \
#   --num-epochs 100 \
#   --batch-size 64 \
#   --scheduler-patience 5 \
#   --scheduler-min-lr 1e-5 \
#   --seed "${seed}" \
#   --timestamp "${current_time}_norc" \
#   --output-csv ${output_dir}/${current_time}_norc_atac_to_5hmc_results.csv \
#   --output-json ${output_dir}/${current_time}_norc_atac_to_5hmc_results.json \
#   --prediction-signal-csv ${output_dir}/${current_time}_norc_atac_to_5hmc_{sample_size}.csv \
#   --regression-plot-path ${output_dir}/${current_time}_norc_atac_to_5hmc_{sample_size}.png

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
echo "[with RC augmentation] timestamp=${current_time}"
python run_multimodal_track_experiments.py \
  --sample-sizes 5000 20000 \
  --input-modality atac \
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
  --scheduler-min-lr 1e-5 \
  --seed "${seed}" \
  --augment-reverse-complement \
  --timestamp "${current_time}_rc" \
  --output-csv ${output_dir}/${current_time}_rc_atac_to_5hmc_results.csv \
  --output-json ${output_dir}/${current_time}_rc_atac_to_5hmc_results.json \
  --prediction-signal-csv ${output_dir}/${current_time}_rc_atac_to_5hmc_{sample_size}.csv \
  --regression-plot-path ${output_dir}/${current_time}_rc_atac_to_5hmc_{sample_size}.png

echo "Done. Compare the two result JSON files under ${output_dir}."