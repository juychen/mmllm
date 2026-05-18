#!/bin/bash
set -e

source /home/zhangyr/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /data1st2/zhangyr/code/mmllm_cjy/mmllm || exit 1

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0"
  echo "This script runs PFC/HIP, MC/MW, target_length 2048/4096/8192 automatically."
  exit 0
fi

regions=("PFC" "HIP")
conditions=("MC" "MW")
target_lengths=("1024" "2048" "4096" "8192")

sample_sizes="2000 10000 20000 50000 100000"

python_script="run_multimodal_track_experiments.py"

for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    for target_length in "${target_lengths[@]}"; do

      output_dir="/data1st2/zhangyr/data/mmllm/test_results/target_length_${target_length}/${region}_${condition}"
      mkdir -p "$output_dir"

      m5c_path="/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz"
      hm5c_path="/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz"
      atac_path="/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw"

      echo "============================================================"
      echo "Running region=${region}, condition=${condition}, target_length=${target_length}"
      echo "m5c:  ${m5c_path}"
      echo "5hmC: ${hm5c_path}"
      echo "ATAC: ${atac_path}"
      echo "Output dir: ${output_dir}"
      echo "============================================================"

      current_time=$(date "+%Y-%m-%d-%H-%M-%S")
      echo "[${region} ${condition} target_length=${target_length}] ATAC -> 5hmC, time: ${current_time}"

      python "$python_script" \
        --sample-sizes $sample_sizes \
        --input-modality atac \
        --augment-reverse-complement \
        --context-modalities sequence \
        --target-modality 5hmc \
        --mask-mode cpg_forward \
        --target-length "$target_length" \
        --m5c-bedgraph "$m5c_path" \
        --hm5c-bedgraph "$hm5c_path" \
        --atac-bw "$atac_path" \
        --scheduler cosine \
        --num-epochs 100 \
        --batch-size 64 \
        --scheduler-patience 5 \
        --timestamp "$current_time" \
        --scheduler-min-lr 1e-5 \
        --output-csv "${output_dir}/${current_time}_atac_to_5hmc_results.csv" \
        --output-json "${output_dir}/${current_time}_atac_to_5hmc_results.json" \
        --prediction-signal-csv "${output_dir}/${current_time}_atac_to_5hmc_{sample_size}.csv" \
        --regression-plot-path "${output_dir}/${current_time}_atac_to_5hmc_{sample_size}.png" \
        --best-checkpoint-path "${output_dir}/${current_time}_atac_to_5hmc_best_{sample_size}.pt" \
        --last-checkpoint-path "${output_dir}/${current_time}_atac_to_5hmc_last_{sample_size}.pt"


      current_time=$(date "+%Y-%m-%d-%H-%M-%S")
      echo "[${region} ${condition} target_length=${target_length}] ATAC -> 5mC, time: ${current_time}"

      python "$python_script" \
        --sample-sizes $sample_sizes \
        --input-modality atac \
        --augment-reverse-complement \
        --context-modalities sequence \
        --target-modality 5mc \
        --mask-mode cpg_forward \
        --target-length "$target_length" \
        --m5c-bedgraph "$m5c_path" \
        --hm5c-bedgraph "$hm5c_path" \
        --atac-bw "$atac_path" \
        --scheduler cosine \
        --num-epochs 100 \
        --batch-size 64 \
        --scheduler-patience 5 \
        --timestamp "$current_time" \
        --scheduler-min-lr 1e-5 \
        --output-csv "${output_dir}/${current_time}_atac_to_5mc_results.csv" \
        --output-json "${output_dir}/${current_time}_atac_to_5mc_results.json" \
        --prediction-signal-csv "${output_dir}/${current_time}_atac_to_5mc_{sample_size}.csv" \
        --regression-plot-path "${output_dir}/${current_time}_atac_to_5mc_{sample_size}.png" \
        --best-checkpoint-path "${output_dir}/${current_time}_atac_to_5mc_best_{sample_size}.pt" \
        --last-checkpoint-path "${output_dir}/${current_time}_atac_to_5mc_last_{sample_size}.pt"


      current_time=$(date "+%Y-%m-%d-%H-%M-%S")
      echo "[${region} ${condition} target_length=${target_length}] 5mC -> 5hmC, time: ${current_time}"

      python "$python_script" \
        --sample-sizes $sample_sizes \
        --input-modality 5mc \
        --augment-reverse-complement \
        --context-modalities sequence \
        --target-modality 5hmc \
        --mask-mode cpg_forward \
        --target-length "$target_length" \
        --m5c-bedgraph "$m5c_path" \
        --hm5c-bedgraph "$hm5c_path" \
        --atac-bw "$atac_path" \
        --scheduler cosine \
        --num-epochs 100 \
        --batch-size 64 \
        --scheduler-patience 5 \
        --timestamp "$current_time" \
        --scheduler-min-lr 1e-5 \
        --output-csv "${output_dir}/${current_time}_5mc_to_5hmc_results.csv" \
        --output-json "${output_dir}/${current_time}_5mc_to_5hmc_results.json" \
        --prediction-signal-csv "${output_dir}/${current_time}_5mc_to_5hmc_{sample_size}.csv" \
        --regression-plot-path "${output_dir}/${current_time}_5mc_to_5hmc_{sample_size}.png" \
        --best-checkpoint-path "${output_dir}/${current_time}_5mc_to_5hmc_best_{sample_size}.pt" \
        --last-checkpoint-path "${output_dir}/${current_time}_5mc_to_5hmc_last_{sample_size}.pt"


      current_time=$(date "+%Y-%m-%d-%H-%M-%S")
      echo "[${region} ${condition} target_length=${target_length}] 5hmC -> 5mC, time: ${current_time}"

      python "$python_script" \
        --sample-sizes $sample_sizes \
        --input-modality 5hmc \
        --augment-reverse-complement \
        --context-modalities sequence \
        --target-modality 5mc \
        --mask-mode cpg_forward \
        --target-length "$target_length" \
        --m5c-bedgraph "$m5c_path" \
        --hm5c-bedgraph "$hm5c_path" \
        --atac-bw "$atac_path" \
        --scheduler cosine \
        --num-epochs 100 \
        --batch-size 64 \
        --scheduler-patience 5 \
        --timestamp "$current_time" \
        --scheduler-min-lr 1e-5 \
        --output-csv "${output_dir}/${current_time}_5hmc_to_5mc_results.csv" \
        --output-json "${output_dir}/${current_time}_5hmc_to_5mc_results.json" \
        --prediction-signal-csv "${output_dir}/${current_time}_5hmc_to_5mc_{sample_size}.csv" \
        --regression-plot-path "${output_dir}/${current_time}_5hmc_to_5mc_{sample_size}.png" \
        --best-checkpoint-path "${output_dir}/${current_time}_5hmc_to_5mc_best_{sample_size}.pt" \
        --last-checkpoint-path "${output_dir}/${current_time}_5hmc_to_5mc_last_{sample_size}.pt"

    done
  done
done
