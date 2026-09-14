#!/bin/bash
# Resume the AMY_MC / AMY_MW modality-dropout runs from their BEST checkpoints.
#
# Uses --resume-from-checkpoint: restores model weights + optimizer state
# (including the learning rate) + LR-scheduler progress + epoch, then continues
# from the next epoch. Hyperparameters mirror the original run exactly (they were
# read back from the checkpoint's saved `args`), so the val split is identical
# and the resulting val_loss is directly comparable.
#
# Outputs get a NEW timestamp and a `_resume` label — nothing is overwritten.
#
# Usage:
#   ./run_all_m5c_query_26_dropmod_resume.sh [fusion_type] [seq_drop_p] [atac_drop_p] [rna_drop_p]
#   CKPT_EVERY=5 ./run_all_m5c_query_26_dropmod_resume.sh        # periodic ckpt every 5 epochs
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"
seq_drop_p="${2:-0.0}"
atac_drop_p="${3:-0.2}"
rna_drop_p="${4:-0.2}"
ckpt_every="${CKPT_EVERY:-5}"

regions=("AMY")
conditions=("MC" "MW")

describe_ckpt() {
  python - "$1" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = ck.get("metrics", {}) or {}
opt = ck.get("optimizer_state_dict") or {}
lr = opt.get("param_groups", [{}])[0].get("lr")
print(f"      epoch={ck.get('epoch')}  val_loss={m.get('val_loss')}  "
      f"val_pearsonr={m.get('val_pearsonr')}  LR={lr}  "
      f"sched_last_epoch={(ck.get('scheduler_state_dict') or {}).get('last_epoch')}")
PY
}

run_resume() {
  local region="$1"
  local condition="$2"
  local fusion="$3"

  local output_dir="output/${region}_${condition}/cCRE_cpg"
  local ckpt
  ckpt="$(ls -t "${output_dir}"/*best*.pt 2>/dev/null | head -1)"
  if [[ -z "$ckpt" ]]; then
    echo "[${region}_${condition}] no *best*.pt in ${output_dir} — skipping."
    return 0
  fi

  local current_time
  current_time=$(date "+%Y-%m-%d-%H-%M-%S")
  local run_label="m5c_query_crosshyena_modelb_${fusion}_dropmod_s${seq_drop_p}_a${atac_drop_p}_r${rna_drop_p}_resume"
  mkdir -p "$output_dir"
  local log_file="${output_dir}/${current_time}_${run_label}.log"

  echo "[$(date)] [${region}_${condition}] RESUME from $(basename "$ckpt")" | tee -a "$log_file"
  echo "      checkpoint contents:" | tee -a "$log_file"
  describe_ckpt "$ckpt" | tee -a "$log_file"

  python run_m5c_query_crosshyena_dropmod.py \
    --resume-from-checkpoint "$ckpt" \
    --sample-sizes all \
    --dmr-csv /data2st1/junyi/generegion_vM23/cCRE_cpg.bed \
    --model-name model_b \
    --model-b-blocks 2 \
    --model-b-fusion "$fusion" \
    --augment-reverse-complement \
    --mask-mode cpg_forward \
    --scheduler cosine \
    --scheduler-min-lr 1e-5 \
    --scheduler-t-max 0 \
    --num-epochs 100 \
    --batch-size 4 \
    --target-length 16384 \
    --gradient-accumulation-steps 64 \
    --scheduler-patience 15 \
    --patience 15 \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --train-ratio 0.8 \
    --seed 7 \
    --amp \
    --gradient-checkpointing \
    --lazy \
    --checkpoint-every-n-epochs "$ckpt_every" \
    --seq-drop-p "$seq_drop_p" \
    --atac-drop-p "$atac_drop_p" \
    --rna-drop-p "$rna_drop_p" \
    --timestamp "$current_time" \
    --m5c-bedgraph "/data8/junyi/methdata/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data8/junyi/methdata/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data8/junyi/methdata/atac/${region}_${condition}_track.bw" \
    --rna-coverage-bw "/data8/junyi/methdata/bulk_rna/BULK_${region}/${region}_${condition}.bw" \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
    --prediction-signal-h5ad "${output_dir}/${current_time}_${run_label}_{sample_size}.h5ad" \
    --regression-plot-path "${output_dir}/${current_time}_${run_label}_{sample_size}.png" \
    --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
    --periodic-checkpoint-path "${output_dir}/${current_time}_${run_label}_periodic_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -eq 0 ]; then
    echo "[$(date)] [${region}_${condition}] resume finished successfully." | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] resume FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

MAX_CONCURRENT=2

echo ""
echo "============================================"
echo "[$(date)] Resuming modality-dropout runs from best checkpoints..."
echo "  fusion=${fusion_type}, drops: seq=${seq_drop_p} atac=${atac_drop_p} rna=${rna_drop_p}, periodic every ${ckpt_every} epoch(s)"
echo "============================================"

failed=0
total=0
running=0

# Optional single-group selection: ONLY=AMY_MW ./run_all_m5c_query_26_dropmod_resume.sh
only="${ONLY:-}"
job_list=()
for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    if [[ -n "$only" && "${region}_${condition}" != "$only" ]]; then
      continue
    fi
    job_list+=("${region}:${condition}")
  done
done
total_jobs=${#job_list[@]}

for job in "${job_list[@]}"; do
  region="${job%%:*}"
  condition="${job##*:}"
  if [ "$running" -ge "$MAX_CONCURRENT" ]; then
    wait -n
    running=$((running - 1))
  fi
  run_resume "$region" "$condition" "$fusion_type" &
  running=$((running + 1))
  total=$((total + 1))
  # Stagger launches: get_freest_gpu() queries nvidia-smi globally, so without a
  # delay consecutive jobs may pick the same GPU (the original runs did, and
  # that contention is why they took ~4 days for 28 epochs).
  if [ "$total" -lt "$total_jobs" ]; then
    sleep "${GPU_STAGGER:-90}"
  fi
done

echo "[$(date)] Waiting for the last ${running} job(s) to complete..."
for job in $(jobs -p); do
  wait "$job" || { failed=$((failed + 1)); }
done

echo ""
echo "============================================"
if [ $failed -eq 0 ]; then
  echo "[$(date)] All ${total} resume run(s) completed successfully!"
else
  echo "[$(date)] $failed out of ${total} resume run(s) FAILED!"
fi
echo "============================================"
