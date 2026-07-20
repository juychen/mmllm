#!/bin/bash
# Run the two missing ATAC ablation experiments.
#
# Ablations (all train on AMY_MC, predict 5hmC):
#   seq_only    : query=DNA,           context=[]          (DNA only — no m5c, no atac)
#   all_three   : query=concat(5mC,ATAC), context=[]       (m5c+atac concat query + DNA via dedicated track)
#
# NOTE on all_three: the original definition was ("m5c_atac", ["sequence"]), but
# FlexibleQueryRegressorModelB hardcodes context_track_dim=1, so sequence (dim=4)
# cannot be a context track without surgery on models.py. We collapse it to
# ("m5c_atac", []) — all three modalities are still consumed (m5c+atac via the
# concat query, DNA via the dedicated sequence_track), which preserves the
# "all three used" intent.
#
# Hyperparameters mirror run_all_atac_ablations.sh exactly so results are
# directly comparable to the full_mirror_modelb sweep.
#
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

# ---- config ----
ABLATIONS=(
  "seq_only"
  "all_three"
)

DATASET_FLAGS=(
  --dmr-csv /data2st1/junyi/generegion_vM23/cCRE_cpg.bed
  --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz
  --atac-bw /data1st1/junyi/methdata/atac/AMY_MC_track.bw
  --genome-fasta /data2st1/junyi/ref/GRCm38.p6.genome.fa
)

TRAINING_FLAGS=(
  --sample-sizes all
  --target-length 16384
  --batch-size 4
  --gradient-accumulation-steps 64
  --mask-mode cpg_forward
  --augment-reverse-complement
  --num-epochs 100
  --scheduler-patience 15
  --patience 15
  --scheduler cosine
  --learning-rate 1e-3
  --weight-decay 1e-5
  --train-ratio 0.8
  --amp
  --gradient-checkpointing
  --lazy
  --hidden-dim 64
  --model-b-blocks 2
  --model-b-fusion cross_hyena
)

# Output lives in a separate subdir so we don't mix these full-data runs with
# the earlier small-sample (5000) experiments. Summary tsv + logs are namespaced
# under RUN_TAG.
RUN_TAG="seq_only_all_three"
RUN_DIR="output/atac_ablation/${RUN_TAG}"
LOG_DIR="logs/atac_ablation_${RUN_TAG}"
mkdir -p "$RUN_DIR" "$LOG_DIR"
summary_file="${RUN_DIR}/_summary.tsv"
echo -e "ablation\tquery\tcontext\tbest_val_loss\tbest_val_r2\tbest_val_pearson" > "$summary_file"

MAX_CONCURRENT=3
running=0
total=0
failed=0

# Wait until there are fewer than MAX_CONCURRENT jobs running, then launch the
# next one. Each job's stdout/stderr is captured to its own log file in
# $LOG_DIR. Jobs are detached with `&` so we can keep launching while others
# are training.
wait_for_slot() {
  while [ "$running" -ge "$MAX_CONCURRENT" ]; do
    wait -n
    running=$((running - 1))
  done
}

launch_ablation() {
  local ablation="$1"
  local log_file="${LOG_DIR}/${ablation}.log"

  echo ""
  echo "============================================"
  echo "[$(date)] Launching ablation: $ablation (log: $log_file)"
  echo "============================================"

  # Train with output-dir under the run-tagged directory (not the shared
  # output/atac_ablation/<ablation>/ from previous small-sample runs).
  python run_atac_ablation.py \
    --ablation "$ablation" \
    --output-dir "${RUN_DIR}" \
    "${DATASET_FLAGS[@]}" \
    "${TRAINING_FLAGS[@]}" \
    > "$log_file" 2>&1

  echo "[$(date)] FINISHED: $ablation (exit=$?)" >> "$log_file"
}

for ablation in "${ABLATIONS[@]}"; do
  wait_for_slot
  launch_ablation "$ablation" &
  running=$((running + 1))
  total=$((total + 1))
done

# Drain the rest.
echo ""
echo "[$(date)] All ${total} ablations queued. Waiting for last ${running} job(s)..."
while [ "$running" -gt 0 ]; do
  wait -n
  running=$((running - 1))
done

# Collect metrics from each ablation's JSON, in original ABLATIONS order.
for ablation in "${ABLATIONS[@]}"; do
  metrics_file=$(ls -t "${RUN_DIR}/${ablation}/"*.json 2>/dev/null | head -1)
  if [[ -f "$metrics_file" ]]; then
    python -c "
import json
with open('$metrics_file') as f:
    m = json.load(f)
print(f'  {m[\"ablation\"]}\\t{m[\"query_modality\"]}\\t{\",\".join(m[\"context_modalities\"])}\\t{m[\"best_val_loss\"]:.4f}\\t{m[\"best_val_r2\"]:.4f}\\t{m[\"best_val_pearson\"]:.4f}', flush=True)
" >> "$summary_file"
  else
    echo "  ✗ $ablation FAILED (no metrics json)" | tee -a "$summary_file"
    failed=$((failed + 1))
  fi
done

echo ""
echo "============================================"
echo "[$(date)] All ablations complete. ${failed} failed of ${total}."
echo ""
echo "Summary (logs: ${LOG_DIR}/):"
column -t -s $'\t' "$summary_file"
echo "============================================"