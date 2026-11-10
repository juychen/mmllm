#!/bin/bash
# Six-group pooled cCRE pretraining, bp0 protocol (no buffer), from scratch.
# Cloned from the 2026-11-03-15-05-59 pooled6 bp50 run; only the mask changes:
#   --mask-neighborhood-bp 0  (hide the CpG's own C/G, nothing around it)
#   --train-loss-mask hide_cpg (loss on every hidden C — the new definition)
#
# Usage:
#   GPU=4 ./run_pooled6_bp0_scratch.sh
#   GPU=4 NUM_EPOCHS=1 SAMPLE_SIZES=64 OUT_DIR=/tmp/p6bp0_smoke ./run_pooled6_bp0_scratch.sh

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1

DMR="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
FASTA="/data2st1/junyi/ref/GRCm38.p6.genome.fa"
M5C=(
  /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MW_AMY.CG.m.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MC_HIP.CG.m.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MW_HIP.CG.m.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.m.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MW_PFC.CG.m.bedGraph.gz
)
HM5C=(
  /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MW_AMY.CG.h.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MC_HIP.CG.h.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MW_HIP.CG.h.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.h.bedGraph.gz
  /data2st1/junyi/output/llm0401/processed_meth/MW_PFC.CG.h.bedGraph.gz
)
ATAC=(
  /data1st1/junyi/methdata/atac/AMY_MC_track.bw
  /data1st1/junyi/methdata/atac/AMY_MW_track.bw
  /data1st1/junyi/methdata/atac/HIP_MC_track.bw
  /data1st1/junyi/methdata/atac/HIP_MW_track.bw
  /data1st1/junyi/methdata/atac/PFC_MC_track.bw
  /data1st1/junyi/methdata/atac/PFC_MW_track.bw
)
RNA=(
  /data2st1/junyi/output/sn0615/BULK_AMY/AMY_MC.bw
  /data2st1/junyi/output/sn0615/BULK_AMY/AMY_MW.bw
  /data2st1/junyi/output/sn0615/BULK_HIP/HIP_MC.bw
  /data2st1/junyi/output/sn0615/BULK_HIP/HIP_MW.bw
  /data2st1/junyi/output/sn0615/BULK_PFC/PFC_MC.bw
  /data2st1/junyi/output/sn0615/BULK_PFC/PFC_MW.bw
)

OUT_DIR="${OUT_DIR:-output/ALL_GROUPS/cCRE_cpg_bp0/scratch30ep}"
LOG_DIR="/home/junyichen/logs"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="$LOG_DIR/pooled6_bp0_scratch_${TS}.out"
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in "$DMR" "$FASTA" "${M5C[@]}" "${HM5C[@]}" "${ATAC[@]}" "${RNA[@]}"; do
  [[ -s "$f" ]] || { echo "[error] missing: $f" >&2; exit 1; }
done

echo "[$(date)] pooled6 bp0 SCRATCH | epochs=${NUM_EPOCHS:-30} | mask_bp=0 | loss=hide_cpg | GPU=${GPU:-4}" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${GPU:-4}" python -u run_multiquery_pretraining.py \
  --sample-sizes "${SAMPLE_SIZES:-all}" \
  --use-all-input-groups \
  --dmr-csv "$DMR" \
  --genome-fasta "$FASTA" \
  --m5c-bedgraph "${M5C[@]}" \
  --hm5c-bedgraph "${HM5C[@]}" \
  --atac-bw "${ATAC[@]}" \
  --rna-coverage-bw "${RNA[@]}" \
  --target-length 16384 \
  --num-blocks 2 \
  --hidden-dim 64 \
  --fusion-type cross_hyena \
  --query-tracks 5mc 5hmc \
  --context-mode all-others \
  --mask-fraction 0.15 \
  --mask-neighborhood-bp 0 \
  --mask-mode cpg_forward \
  --train-loss-mask hide_cpg \
  --drop-p-5mc 0.0 \
  --drop-p-5hmc 0.0 \
  --drop-p-atac 0.1 \
  --drop-p-rna 0.1 \
  --drop-p-seq 0.0 \
  --scheduler cosine \
  --num-epochs "${NUM_EPOCHS:-30}" \
  --learning-rate 1e-3 \
  --scheduler-min-lr 1e-5 \
  --warmup-epochs 0 \
  --batch-size 32 \
  --gradient-accumulation-steps 8 \
  --num-workers 8 \
  --patience 15 \
  --amp \
  --augment-reverse-complement \
  --no-export-test-h5ad \
  --seed 7 \
  --output-csv "$OUT_DIR/${TS}_pooled6_2heads_s0.0_a0.1_r0.1_cpgpair_bp0_lossc_results.csv" \
  2>&1 | tee -a "$LOG"