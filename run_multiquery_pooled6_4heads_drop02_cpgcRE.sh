#!/bin/bash
# Pooled 6-group training with the CURRENT multi-query model — FOUR heads.
#
#   groups   : AMY / HIP / PFC  x  MC / MW  = 6 input groups, ONE pooled model
#   queries  : 5mC + 5hmC + ATAC + RNA, each its own stream + output head
#   context  : DNA sequence only (context_tracks=[]); all-others: each head
#              also sees the OTHER three query tracks
#   dropout  : ATAC 0.2, RNA 0.2 (modality dropout on the ATAC/RNA query tracks);
#              5mC/5hmC/seq = 0
#   regions  : cCRE_cpg.bed (CpG cCRE), 80,137 windows per group -> 480,822 total
#
# The 6 groups are pooled with --use-all-input-groups: the DMR table is
# replicated once per group, and each row carries its own region/condition
# track paths. One epoch therefore covers 6 x 80,137 = 480,822 windows.
#
# Usage:
#   ./run_multiquery_pooled6_4heads_drop02_cpgcRE.sh
#   NUM_EPOCHS=15 GPU=0 ./run_multiquery_pooled6_4heads_drop02_cpgcRE.sh
#   N_SMOKE=1 GPU=0 ./run_multiquery_pooled6_4heads_drop02_cpgcRE.sh   # 64x6 rows, 1 epoch
#
# Env:
#   NUM_EPOCHS=30    BATCH_SIZE=16   GRAD_ACCUM=16   NUM_WORKERS=8
#   GPU=0            LOG_DIR=/home/junyichen/logs
#   OUT_ROOT=output/ALL_GROUPS/cCRE_cpg
#   RUN_TAG=""       extra label appended to the run name
#
# Runtime: 480,822 windows is ~6x the single-group cCRE run. The 4-head model is
# ~2x the 2-head cost, so one pooled epoch is ~9-11 h; 30 epochs is ~12-14 days,
# 15 epochs ~6-7 days. Set NUM_EPOCHS=15 for a first pooled 4-head run.

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1
export PYTHONUNBUFFERED=1

dmr_csv="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
bed_name="$(basename "$dmr_csv" .bed)"
M="/data2st1/junyi/output/llm0401/processed_meth"
A="/data1st1/junyi/methdata/atac"
R="/data2st1/junyi/output/sn0615"

num_epochs="${NUM_EPOCHS:-30}"
batch_size="${BATCH_SIZE:-16}"
grad_accum="${GRAD_ACCUM:-16}"
num_workers="${NUM_WORKERS:-8}"
sample_sizes="${SAMPLE_SIZES:-all}"
gpu="${GPU:-0}"
run_tag="${RUN_TAG:-}"
log_dir="${LOG_DIR:-/home/junyichen/logs}"
out_root="${OUT_ROOT:-output/ALL_GROUPS/${bed_name}}"

# ---- pooled groups, fixed order ---------------------------------------------
regions=(AMY AMY HIP HIP PFC PFC)
conditions=(MC MW MC MW MC MW)
m5c_paths=(); hm5c_paths=(); atac_paths=(); rna_paths=()
for i in "${!regions[@]}"; do
  r="${regions[$i]}"; c="${conditions[$i]}"
  m5c_paths+=("$M/${c}_${r}.CG.m.bedGraph.gz")
  hm5c_paths+=("$M/${c}_${r}.CG.h.bedGraph.gz")
  atac_paths+=("$A/${r}_${c}_track.bw")
  rna_paths+=("$R/BULK_${r}/${r}_${c}.bw")
done

for f in "${m5c_paths[@]}" "${hm5c_paths[@]}" "${atac_paths[@]}" "${rna_paths[@]}"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done

# ---- smoke mode --------------------------------------------------------------
if [[ -n "${N_SMOKE:-}" ]]; then
  num_epochs=1
  batch_size=8
  grad_accum=1
  num_workers=0
  sample_sizes=64
  out_root="/tmp/mq_pooled6_4heads_smoke"
  log_dir="/tmp"
  echo "[smoke] 64 rows x 6 groups, 1 epoch, batch 8"
fi

mkdir -p "$out_root" "$log_dir"
ts="$(date +%Y-%m-%d-%H-%M-%S)"
run_label="multiquery_pooled6_4heads_s0.0_a0.2_r0.2${run_tag:+_${run_tag}}"
log_file="${log_dir}/${ts}_${run_label}.out"

echo "[$(date)] pooled 6-group 4-head training" | tee -a "$log_file"
echo "  groups   : ${regions[*]} x ${conditions[*]} (6)" | tee -a "$log_file"
echo "  queries  : 5mc 5hmc atac rna (4 heads), context all-others" | tee -a "$log_file"
echo "  dropout  : atac=0.2 rna=0.2 (5mc=0 5hmc=0 seq=0)" | tee -a "$log_file"
echo "  regions  : $dmr_csv" | tee -a "$log_file"
echo "  epochs   : $num_epochs | batch ${batch_size} x accum ${grad_accum} | GPU $gpu" | tee -a "$log_file"
echo "  output   : $out_root" | tee -a "$log_file"

CUDA_VISIBLE_DEVICES="$gpu" python run_multiquery_pretraining.py \
  --sample-sizes "$sample_sizes" \
  --use-all-input-groups \
  --dmr-csv "$dmr_csv" \
  --target-length 16384 \
  --num-blocks 2 \
  --hidden-dim 64 \
  --fusion-type cross_hyena \
  --query-tracks 5mc 5hmc atac rna \
  --context-mode all-others \
  --mask-fraction 0.15 \
  --mask-mode cpg_forward \
  --augment-reverse-complement \
  --drop-p-5mc 0.0 --drop-p-5hmc 0.0 --drop-p-atac 0.2 --drop-p-rna 0.2 --drop-p-seq 0.0 \
  --scheduler cosine \
  --num-epochs "$num_epochs" \
  --batch-size "$batch_size" \
  --gradient-accumulation-steps "$grad_accum" \
  --num-workers "$num_workers" \
  --scheduler-patience 15 \
  --patience 15 \
  --amp \
  --timestamp "$ts" \
  --scheduler-min-lr 1e-5 \
  --m5c-bedgraph "${m5c_paths[@]}" \
  --hm5c-bedgraph "${hm5c_paths[@]}" \
  --atac-bw "${atac_paths[@]}" \
  --rna-coverage-bw "${rna_paths[@]}" \
  --output-csv "${out_root}/${ts}_${run_label}_results.csv" \
  --output-json "${out_root}/${ts}_${run_label}_results.json" \
  --best-checkpoint-path "${out_root}/${ts}_${run_label}_best_{sample_size}.pt" \
  --last-checkpoint-path "${out_root}/${ts}_${run_label}_last_{sample_size}.pt" \
  2>&1 | tee -a "$log_file"

code=${PIPESTATUS[0]}
if [[ $code -eq 0 ]]; then
  echo "[$(date)] finished" | tee -a "$log_file"
else
  echo "[$(date)] FAILED (exit $code)" | tee -a "$log_file"
fi
exit "$code"
