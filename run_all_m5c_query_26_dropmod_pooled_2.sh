#!/bin/bash
# Pooled multi-dataset modality-dropout training: ONE model over all 3 brain
# regions (AMY/HIP/PFC) x 2 conditions (MC/MW) = 6 input groups.
#
# Uses --use-all-input-groups with per-row track paths (now supported by the
# lazy datasets), so every DMR window reads its OWN region/condition tracks.
#
# Usage:
#   ./run_all_m5c_query_26_dropmod_pooled.sh [fusion_type] [seq_drop_p] [atac_drop_p] [rna_drop_p]
# Example:
#   ./run_all_m5c_query_26_dropmod_pooled.sh cross_hyena 0.0 0.2 0.2
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"
seq_drop_p="${2:-0.0}"
atac_drop_p="${3:-0.2}"
rna_drop_p="${4:-0.2}"

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported fusion: $fusion_type"; exit 1
fi

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
bed_name="$(basename "$dmr_csv" .bed)"

# Optional strict continuation:
#   RESUME_CKPT=/path/to/checkpoint.pt ./run_all_m5c_query_26_dropmod_pooled.sh
# Restores model + optimizer (incl. LR) + scheduler + epoch and resumes there.
resume_tag=""
resume_args=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
  resume_tag="_resume"
  resume_args=(--resume-from-checkpoint "$RESUME_CKPT")
fi

# Periodic checkpoint every N epochs (protects the ~25h pooled run). 0 disables.
ckpt_every="${CKPT_EVERY:-5}"

run_label="m5c_query_crosshyena_modelb_${fusion_type}_dropmod_pooled6_s${seq_drop_p}_a${atac_drop_p}_r${rna_drop_p}${resume_tag}"
output_dir="output/ALL_GROUPS/${bed_name}"
mkdir -p "$output_dir"
log_file="${output_dir}/${current_time}_${run_label}.log"

M=/data8/junyi/methdata/processed_meth
A=/data8/junyi/methdata/atac
R=/data8/junyi/methdata/bulk_rna

echo "[$(date)] Pooled 6-group training: AMY/HIP/PFC x MC/MW" | tee -a "$log_file"
echo "[$(date)] drops: seq=${seq_drop_p} atac=${atac_drop_p} rna=${rna_drop_p}" | tee -a "$log_file"
echo "[$(date)] periodic checkpoint every ${ckpt_every} epoch(s); resume=${RESUME_CKPT:-none}" | tee -a "$log_file"
echo "[$(date)] log: ${log_file}" | tee -a "$log_file"

python run_m5c_query_crosshyena_dropmod.py \
  --sample-sizes all \
  --use-all-input-groups \
  "${resume_args[@]}" \
  --checkpoint-every-n-epochs "$ckpt_every" \
  --dmr-csv "$dmr_csv" \
  --model-name model_b \
  --model-b-blocks 2 \
  --model-b-fusion "$fusion_type" \
  --augment-reverse-complement \
  --mask-mode cpg_forward \
  --scheduler cosine \
  --num-epochs 100 \
  --batch-size 4 \
  --target-length 16384 \
  --gradient-accumulation-steps 64 \
  --scheduler-patience 15 \
  --patience 15 \
  --amp \
  --gradient-checkpointing \
  --lazy \
  --timestamp "$current_time" \
  --scheduler-min-lr 1e-5 \
  --seq-drop-p "$seq_drop_p" \
  --atac-drop-p "$atac_drop_p" \
  --rna-drop-p "$rna_drop_p" \
  --m5c-bedgraph \
    "$M/MC_AMY.CG.m.bedGraph.gz" "$M/MW_AMY.CG.m.bedGraph.gz" \
    "$M/MC_HIP.CG.m.bedGraph.gz" "$M/MW_HIP.CG.m.bedGraph.gz" \
    "$M/MC_PFC.CG.m.bedGraph.gz" "$M/MW_PFC.CG.m.bedGraph.gz" \
  --hm5c-bedgraph \
    "$M/MC_AMY.CG.h.bedGraph.gz" "$M/MW_AMY.CG.h.bedGraph.gz" \
    "$M/MC_HIP.CG.h.bedGraph.gz" "$M/MW_HIP.CG.h.bedGraph.gz" \
    "$M/MC_PFC.CG.h.bedGraph.gz" "$M/MW_PFC.CG.h.bedGraph.gz" \
  --atac-bw \
    "$A/AMY_MC_track.bw" "$A/AMY_MW_track.bw" \
    "$A/HIP_MC_track.bw" "$A/HIP_MW_track.bw" \
    "$A/PFC_MC_track.bw" "$A/PFC_MW_track.bw" \
  --rna-coverage-bw \
    "$R/BULK_AMY/AMY_MC.bw" "$R/BULK_AMY/AMY_MW.bw" \
    "$R/BULK_HIP/HIP_MC.bw" "$R/BULK_HIP/HIP_MW.bw" \
    "$R/BULK_PFC/PFC_MC.bw" "$R/BULK_PFC/PFC_MW.bw" \
  --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
  --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
  --prediction-signal-h5ad "${output_dir}/${current_time}_${run_label}_{sample_size}.h5ad" \
  --regression-plot-path "${output_dir}/${current_time}_${run_label}_{sample_size}.png" \
  --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
  --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
  2>&1 | tee -a "$log_file"

code=${PIPESTATUS[0]}
if [ $code -eq 0 ]; then
  echo "[$(date)] Pooled 6-group training finished successfully." | tee -a "$log_file"
else
  echo "[$(date)] Pooled 6-group training FAILED with exit code $code" | tee -a "$log_file"
fi
exit $code
