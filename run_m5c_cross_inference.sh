#!/bin/bash
# Cross-region / cross-condition inference: test a trained model on
# HIP_MC, HIP_MW, PFC_MC, PFC_MW  (and optionally AMY_MW).
#
# Model config (target_length, mask_mode, model architecture, etc.) is
# AUTO-READ from the experiment results.json.
#
# Usage:
#   bash run_m5c_cross_inference.sh                                       # use hardcoded config
#   bash run_m5c_cross_inference.sh <checkpoint.pt> [experiment.json] [dmr.bed]  # override
#
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

# ---- hardcoded config (from VS Code debug config) ----
# Checkpoint: AMY_MC trained, model_b / cross_hyena, 16k, cpg_forward
CHECKPOINT="/data1st1/junyi/output/mmllm/AMY_MC/2026-06-30-15-38-58_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_best_80137.pt"
EXPERIMENT_JSON="/data1st1/junyi/output/mmllm/AMY_MC/2026-06-30-15-38-58_m5c_query_sequence_atac_crosshyena_modelb_cross_hyena_results.json"
DMR_CSV="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
SAMPLE_SIZES=10000

# Allow overriding via positional args (optional)
if [[ -n "$1" ]]; then
  CHECKPOINT="$1"
fi
if [[ -n "$2" ]]; then
  EXPERIMENT_JSON="$2"
fi
if [[ -n "$3" ]]; then
  DMR_CSV="$3"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi
if [[ ! -f "$EXPERIMENT_JSON" ]]; then
  echo "ERROR: experiment json not found: $EXPERIMENT_JSON"
  exit 1
fi

json_flag="--experiment-json $EXPERIMENT_JSON"

echo "============================================"
echo "[$(date)] Cross-region / cross-condition inference"
echo "  Checkpoint: $(basename "$CHECKPOINT")"
echo "  JSON:       $(basename "$EXPERIMENT_JSON")"
echo "  DMR:        $(basename "$DMR_CSV")"
echo "  Samples:    $SAMPLE_SIZES"
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
    --sample-sizes "$SAMPLE_SIZES" \
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
