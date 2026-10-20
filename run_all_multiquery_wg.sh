#!/bin/bash
# Run multi-query masked pretraining for AMY/HIP/PFC x MC/MW — parallel submission.
#
# Calls run_multiquery_pretraining.py:
#   * queries : 5mC AND 5hmC each get their OWN stream (proj+norm) + output head,
#               and each runs through the SHARED StripedHyena blocks.
#   * context : ATAC + RNA + DNA sequence, fused ONCE and shared by both queries.
#   * masking : shared CpG position mask across all signal tracks (learned
#               [POS-MASK] tokens) + optional per-track modality mask.
#   * loss    : computed only at position-masked positions (anti-leak; the two
#               query streams never read each other).
#
# Usage:
#   ./run_all_multiquery_wg.sh [fusion_type] [seq_drop_p] [atac_drop_p] [rna_drop_p]
# Example:
#   ./run_all_multiquery_wg.sh cross_hyena 0.0 0.2 0.2
#
# Optional env vars:
#   TEST_BED=/path/to/test.bed      test BED (default: output/holdout_testchroms/cpg_test.bed)
#                                   --test-bed selects the --dmr-csv windows overlapping it
#   MASK_FRACTION=0.15              shared CpG position-mask fraction
#   BATCH_SIZE=32                   micro-batch (default 32)
#   GRAD_ACCUM=8                    gradient accumulation (default 8; keep BATCH_SIZE*GRAD_ACCUM=256)
#   NUM_WORKERS=8                   dataloader workers (default 8)
#   NUM_EPOCHS=100                  max epochs (default 100)
#   GPUS="1 4 6 7"                  OPTIONAL manual override (round-robin over just these)
#   MIN_FREE_MIB=40000              a GPU qualifies only with >= this much free  (auto mode)
#   MAX_USED_MIB=20000              ...and <= this much already used            (auto mode)
#   GPU_DRYRUN=1                    print the GPU choice and exit (no jobs launched)
#   MAX_CONCURRENT=4                max parallel jobs (auto-capped to the #idle GPUs)
#   OUT_TAG=                        extra label appended to the run label
#
# GPU selection: if GPUS is NOT set, the script picks the IDLEST GPUs itself —
# least-used first, keeping only cards with >= MIN_FREE_MIB free and
# <= MAX_USED_MIB used (so it never lands next to another ~38 GB job).
#
# Tip: smoke-test with  N_SMOKE=1 ./run_all_multiquery_wg.sh   (2000 windows, 2 epochs)
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

fusion_type="${1:-cross_hyena}"
seq_drop_p="${2:-0.0}"
atac_drop_p="${3:-0.0}"
rna_drop_p="${4:-0.0}"

mask_fraction="${MASK_FRACTION:-0.15}"
context_mode="${CONTEXT_MODE:-shared}"
batch_size="${BATCH_SIZE:-32}"
grad_accum="${GRAD_ACCUM:-8}"
num_workers="${NUM_WORKERS:-8}"
num_epochs="${NUM_EPOCHS:-100}"
max_concurrent="${MAX_CONCURRENT:-4}"
out_tag="${OUT_TAG:-}"

# ---- GPU selection (auto unless GPUS is set) --------------------------------
min_free_mib="${MIN_FREE_MIB:-40000}"
max_used_mib="${MAX_USED_MIB:-20000}"

if [[ -n "${GPUS:-}" ]]; then
  gpus=($GPUS)
  echo "[gpu] using GPUS from env: ${gpus[*]}"
else
  # least-used GPUs first; keep only cards that are (nearly) idle AND have room
  mapfile -t gpus < <(
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' ' \
      | awk -F',' -v mf="$min_free_mib" -v mu="$max_used_mib" \
          '{ if ($2+0 <= mu && $3+0 >= mf) print $2, $1 }' \
      | sort -k1,1n -k2,2n \
      | awk '{print $2}'
  )
  if [[ ${#gpus[@]} -eq 0 ]]; then
    echo "ERROR: no idle GPU found (need used <= ${max_used_mib} MiB AND free >= ${min_free_mib} MiB)." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv >&2
    echo "       set GPUS='...' explicitly, or relax MIN_FREE_MIB / MAX_USED_MIB." >&2
    exit 1
  fi
  echo "[gpu] auto-selected idle GPUs (least-used first): ${gpus[*]}"
fi

if [[ "${GPU_DRYRUN:-0}" == "1" ]]; then
  echo "[gpu] dry-run: would use GPUs ${gpus[*]} (max_concurrent=${max_concurrent}); nothing launched."
  exit 0
fi

if [[ "$max_concurrent" -gt "${#gpus[@]}" ]]; then
  echo "[gpu] capping MAX_CONCURRENT ${max_concurrent} -> ${#gpus[@]} (only that many idle GPUs available)"
  max_concurrent="${#gpus[@]}"
fi

if [[ "$fusion_type" != "cross_hyena" && "$fusion_type" != "cross_attention" ]]; then
  echo "Unsupported fusion: $fusion_type"
  echo "Allowed values: cross_hyena cross_attention"
  exit 1
fi

regions=(${REGIONS:-AMY})
conditions=(${CONDITIONS:-MC})

smoke_args=()
sample_sizes="all"
if [[ -n "${N_SMOKE:-}" ]]; then
  sample_sizes="2000"
  num_epochs=2
  echo "[smoke] running a 2000-window, 2-epoch smoke test"
fi

run_experiment() {
  local region="$1"
  local condition="$2"
  local fusion_type="$3"
  local gpu="$4"

  local current_time
  current_time=$(date "+%Y-%m-%d-%H-%M-%S")

  local dmr_csv="/data1st1/junyi/output/mmllm/holdout_testchroms/wg_trainpool_plus_cpgtest.bed"
  local bed_name
  bed_name="$(basename "$dmr_csv" .bed | sed 's/\.bed\.gz//')"

  local run_label="multiquery_pretrain_ctx${context_mode}_s${seq_drop_p}_a${atac_drop_p}_r${rna_drop_p}${out_tag}"
  local output_dir="output/${region}_${condition}/${bed_name}"
  mkdir -p "$output_dir"
  local log_file="${output_dir}/${current_time}_${run_label}.log"

  # Single source of truth for the test set.  NOTE: --test-bed SELECTS, out of
  # --dmr-csv, the windows overlapping this BED — it does not import windows of
  # its own.  That is why --dmr-csv below is the COMBINED file (train pool +
  # these test windows); with a train-only dmr-csv the intersection would be
  # empty and test would be 0.
  local test_bed="${TEST_BED:-output/holdout_testchroms/cpg_test.bed}"
  local test_args=(--test-bed "$test_bed")

  echo "[$(date)] [${region}_${condition}] GPU${gpu} | multi-query masked pretraining (BED: ${bed_name})" | tee -a "$log_file"
  echo "[$(date)] [${region}_${condition}] queries=5mc,5hmc | context_mode=${context_mode} | context=atac,rna,seq | mask_fraction=${mask_fraction} | drops: s=${seq_drop_p} a=${atac_drop_p} r=${rna_drop_p} | bs=${batch_size} accum=${grad_accum} workers=${num_workers}" | tee -a "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" python run_multiquery_pretraining.py \
    --sample-sizes "$sample_sizes" \
    --dmr-csv "$dmr_csv" \
    --target-length 16384 \
    --num-blocks 2 \
    --fusion-type "$fusion_type" \
    --query-tracks 5mc 5hmc \
    --context-mode "$context_mode" \
    --mask-fraction "$mask_fraction" \
    --drop-p-seq "$seq_drop_p" \
    --drop-p-atac "$atac_drop_p" \
    --drop-p-rna "$rna_drop_p" \
    --augment-reverse-complement \
    --mask-mode cpg_forward \
    --scheduler cosine \
    --num-epochs "$num_epochs" \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$grad_accum" \
    --num-workers "$num_workers" \
    --scheduler-patience 15 \
    --patience 15 \
    --amp \
    --timestamp "$current_time" \
    --scheduler-min-lr 1e-5 \
    "${test_args[@]}" \
    --m5c-bedgraph  "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data1st1/junyi/methdata/atac/${region}_${condition}_track.bw" \
    --rna-coverage-bw "/data2st1/junyi/output/sn0615/BULK_${region}/${region}_${condition}.bw" \
    --output-csv "${output_dir}/${current_time}_${run_label}_results.csv" \
    --output-json "${output_dir}/${current_time}_${run_label}_results.json" \
    --best-checkpoint-path "${output_dir}/${current_time}_${run_label}_best_{sample_size}.pt" \
    --last-checkpoint-path "${output_dir}/${current_time}_${run_label}_last_{sample_size}.pt" \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -eq 0 ]; then
    echo "[$(date)] [${region}_${condition}] Finished successfully!" | tee -a "$log_file"
  else
    echo "[$(date)] [${region}_${condition}] FAILED with exit code $exit_code" | tee -a "$log_file"
  fi
  return $exit_code
}

echo ""
echo "============================================"
echo "[$(date)] Submitting multi-query masked pretraining (max ${max_concurrent} concurrent, GPUs: ${gpus[*]})..."
echo "============================================"

failed=0
total=0
running=0
gpu_i=0

for region in "${regions[@]}"; do
  for condition in "${conditions[@]}"; do
    if [ "$running" -ge "$max_concurrent" ]; then
      wait -n
      running=$((running - 1))
    fi
    gpu="${gpus[$((gpu_i % ${#gpus[@]}))]}"
    gpu_i=$((gpu_i + 1))
    run_experiment "$region" "$condition" "$fusion_type" "$gpu" &
    running=$((running + 1))
    total=$((total + 1))
  done
done

echo "[$(date)] Waiting for the last ${running} job(s) to complete..."
for job in $(jobs -p); do
  wait "$job" || { failed=$((failed + 1)); }
done

echo ""
echo "============================================"
if [ $failed -eq 0 ]; then
  echo "[$(date)] All ${total} experiments completed successfully!"
else
  echo "[$(date)] $failed out of ${total} experiments FAILED!"
fi
echo "============================================"
