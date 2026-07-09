#!/bin/bash
# Inference-only: load a trained checkpoint and run predictions.
#
# Usage:
#   bash run_m5c_inference.sh <checkpoint.pt> [REGION] [CONDITION]
#
# Examples:
#   bash run_m5c_inference.sh output/AMY_MC/.../best_500000.pt  AMY MC
#   bash run_m5c_inference.sh output/AMY_MC/.../best_500000.pt  ALL
#
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

CHECKPOINT="${1:?Usage: $0 <checkpoint.pt> [REGION] [CONDITION]}"
region="${2:-AMY}"
condition="${3:-MC}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi

if [[ "$region" != "AMY" && "$region" != "HIP" && "$region" != "PFC" && "$region" != "ALL" ]]; then
  echo "Unsupported region: $region  (allowed: AMY HIP PFC ALL)"
  exit 1
fi

if [[ "$region" != "ALL" && "$condition" != "MC" && "$condition" != "MW" ]]; then
  echo "Unsupported condition: $condition  (allowed: MC MW)"
  exit 1
fi

current_time=$(date "+%Y-%m-%d-%H-%M-%S")
echo "[$(date)] Running inference with checkpoint: $(basename "$CHECKPOINT")"
echo "  Region: $region  Condition: $condition"

if [[ "$region" == "ALL" ]]; then
  python run_m5c_inference.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "output/inference/ALL_GROUPS" \
    --use-all-input-groups \
    --lazy \
    --m5c-bedgraph \
      /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MW_AMY.CG.m.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MC_HIP.CG.m.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MW_HIP.CG.m.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.m.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MW_PFC.CG.m.bedGraph.gz \
    --hm5c-bedgraph \
      /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MW_AMY.CG.h.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MC_HIP.CG.h.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MW_HIP.CG.h.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MC_PFC.CG.h.bedGraph.gz \
      /data2st1/junyi/output/llm0401/processed_meth/MW_PFC.CG.h.bedGraph.gz \
    --atac-bw \
      /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw \
      /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MW_track.bw \
      /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/HIP_MC_track.bw \
      /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/HIP_MW_track.bw \
      /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/PFC_MC_track.bw \
      /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/PFC_MW_track.bw
else
  python run_m5c_inference.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "output/inference/${region}_${condition}" \
    --lazy \
    --m5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.m.bedGraph.gz" \
    --hm5c-bedgraph "/data2st1/junyi/output/llm0401/processed_meth/${condition}_${region}.CG.h.bedGraph.gz" \
    --atac-bw "/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/${region}_${condition}_track.bw"
fi

exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "[$(date)] Inference finished successfully!"
else
  echo "[$(date)] Inference FAILED with exit code $exit_code"
fi
exit $exit_code
