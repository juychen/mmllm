#!/bin/bash
# Warm-up / fine-tuning from a pre-trained downstream checkpoint.
#
# Loads a trained downstream model_b checkpoint and continues training on
# a new set of genomic regions (whole_genome_16kb_noCpG_val.bed), with
# adjusted LR and scheduler for fine-tuning.
#
# Usage:
#   bash run_warmup_m5c_query_sequence_atac_crosshyena.sh

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

# ---- config ----
REGION_BED="/data1st1/junyi/output/mmllm/whole_genome_16kb_noCpG_val.bed"
INIT_CKPT="/data1st1/junyi/output/mmllm/AMY_MC/cCRE_cpg/2026-07-14-14-50-08_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_last_80137.pt"

GENOME_FASTA="/data2st1/junyi/ref/GRCm38.p6.genome.fa"

M5C_BEDGRAPH="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz"
HM5C_BEDGRAPH="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"
ATAC_BW="/data1st1/junyi/methdata/atac/AMY_MC_track.bw"

TARGET_LENGTH=16384
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=64
HIDDEN_DIM=64
MODEL_B_BLOCKS=2
MODEL_B_FUSION="cross_hyena"

# Warm-up specific hyper-parameters
LEARNING_RATE=1e-4
SCHEDULER="cosine"
SCHEDULER_MIN_LR=1e-6
SCHEDULER_T_MAX=0        # 0 = use num_epochs as T_max
NUM_EPOCHS=50
PATIENCE=10

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
output_dir="output/AMY_MC/warmup_whole_genome_nocpg"

echo "============================================"
echo "[$(date)] Warm-up training start"
echo "  Init checkpoint: $INIT_CKPT"
echo "  Region BED:      $REGION_BED"
echo "  Target length:   $TARGET_LENGTH"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Scheduler:       $SCHEDULER (min_lr=$SCHEDULER_MIN_LR)"
echo "  Num epochs:      $NUM_EPOCHS"
echo "  Patience:        $PATIENCE"
echo "  Output dir:      $output_dir"
echo "============================================"

python run_m5c_query_sequence_atac_crosshyena_experiments.py \
  --dmr-csv "$REGION_BED" \
  --genome-fasta "$GENOME_FASTA" \
  --m5c-bedgraph "$M5C_BEDGRAPH" \
  --hm5c-bedgraph "$HM5C_BEDGRAPH" \
  --atac-bw "$ATAC_BW" \
  --sample-sizes all \
  --target-length "$TARGET_LENGTH" \
  --train-ratio 0.8 \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --model-name model_b \
  --model-b-blocks "$MODEL_B_BLOCKS" \
  --model-b-fusion "$MODEL_B_FUSION" \
  --num-epochs "$NUM_EPOCHS" \
  --patience "$PATIENCE" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay 1e-5 \
  --scheduler "$SCHEDULER" \
  --scheduler-min-lr "$SCHEDULER_MIN_LR" \
  --scheduler-t-max "$SCHEDULER_T_MAX" \
  --atac-scaling minmax \
  --mask-mode cpg_forward \
  --augment-reverse-complement \
  --amp \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient-checkpointing \
  --lazy \
  --seed 7 \
  --init-from-checkpoint "$INIT_CKPT" \
  --timestamp "$current_time" \
  --output-csv "${output_dir}/${current_time}_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_results.csv" \
  --output-json "${output_dir}/${current_time}_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_results.json" \
  --prediction-signal-h5ad "${output_dir}/${current_time}_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_{sample_size}.h5ad" \
  --regression-plot-path "${output_dir}/${current_time}_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_{sample_size}.png" \
  --best-checkpoint-path "${output_dir}/${current_time}_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_best_{sample_size}.pt" \
  --last-checkpoint-path "${output_dir}/${current_time}_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_last_{sample_size}.pt"

exit_code=$?

echo ""
echo "============================================"
if [ $exit_code -eq 0 ]; then
  echo "[$(date)] Warm-up training finished successfully!"
else
  echo "[$(date)] Warm-up training FAILED with exit code $exit_code"
fi
echo "============================================"

exit $exit_code
