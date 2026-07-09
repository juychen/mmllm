#!/bin/bash
# Cross-region / cross-condition inference: test a trained model on
# HIP_MC, HIP_MW, PFC_MC, PFC_MW  (and optionally AMY_MW).
#
# Model config (target_length, mask_mode, model architecture, etc.) is
# AUTO-READ from the experiment results.json (auto-detected or user-specified).
#
# Usage:
#   bash run_m5c_cross_inference.sh <checkpoint.pt> [experiment.json] [dmr.bed]
#
# Examples:
#   # Auto-detect json from checkpoint directory
#   bash run_m5c_cross_inference.sh path/to/best_80137.pt
#
#   # Specify json explicitly
#   bash run_m5c_cross_inference.sh path/to/best.pt path/to/experiment_results.json
#
#   # Specify everything
#   bash run_m5c_cross_inference.sh path/to/best.pt path/to/results.json whole_genome_16kb.bed
#
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

# ---- config (positional args or defaults) ----
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt> [experiment.json] [dmr.bed]}"
EXPERIMENT_JSON="${2:-}"        # empty = auto-detect from checkpoint dir
DMR_CSV="${3:-/data2st1/junyi/generegion_vM23/cCRE_cpg.bed}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi

# Build --experiment-json flag if provided
json_flag=""
if [[ -n "$EXPERIMENT_JSON" ]]; then
  if [[ ! -f "$EXPERIMENT_JSON" ]]; then
    echo "ERROR: experiment json not found: $EXPERIMENT_JSON"
    exit 1
  fi
  json_flag="--experiment-json $EXPERIMENT_JSON"
fi

echo "============================================"
echo "[$(date)] Cross-region / cross-condition inference"
echo "  Checkpoint: $(basename "$CHECKPOINT")"
[[ -n "$EXPERIMENT_JSON" ]] && echo "  JSON:       $(basename "$EXPERIMENT_JSON")" || echo "  JSON:       (auto-detect)"
echo "  DMR:        $(basename "$DMR_CSV")"
echo "============================================"

# Regions/conditions to test (excluding AMY_MC which was the training data)
targets=(
  "HIP MC"
  "HIP MW"
  "PFC MC"
  "PFC MW"
  "AMY MW"   # same region, different condition
)

failed=0
total=0

for pair in "${targets[@]}"; do
  read -r region condition <<< "$pair"

  output_subdir="output/inference/cross_test/${region}_${condition}"
  mkdir -p "$output_subdir"

  echo ""
  echo ">>> [$(date)] Testing on ${region}_${condition} ..."

  # shellcheck disable=SC2086  # json_flag is intentionally word-split
  python run_m5c_inference.py \
    --checkpoint "$CHECKPOINT" \
    $json_flag \
    --dmr-csv "$DMR_CSV" \
    --output-dir "$output_subdir" \
    --lazy \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw"

  if [ $? -eq 0 ]; then
    echo "    ${region}_${condition} ✓"
  else
    echo "    ${region}_${condition} ✗ FAILED"
    failed=$((failed + 1))
  fi
  total=$((total + 1))
done

echo ""
echo "============================================"
if [ $failed -eq 0 ]; then
  echo "[$(date)] All ${total} cross-tests completed successfully!"
else
  echo "[$(date)] $failed out of ${total} tests FAILED!"
fi
echo "Results in: output/inference/cross_test/"
echo "============================================"
