#!/bin/bash
# Exact Shapley / interaction analysis for the AMY_MC cCRE checkpoint.
#
# ONE job, ONE GPU, the whole test split (7,412 windows, ~41 min).
# Per window the python script enumerates 2^5 = 32 coalitions (sequence, 5mC,
# ATAC, RNA and the 5hmC neighbourhood) and writes per-CpG phi / interaction.
#
# Usage:
#   ./run_xai_shapley_all.sh              # GPU 5
#   GPU=6 ./run_xai_shapley_all.sh        # another card
#   nohup ./run_xai_shapley_all.sh > /dev/null 2>&1 &
#
# Output: output/AMY_MC/cCRE_cpg/xai/test_5hmc/
# Log:    logs/xai_shapley_amy_mc_<timestamp>.log

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1
export PYTHONUNBUFFERED=1

CKPT="/data1st1/junyi/output/mmllm/AMY_MC/cCRE_cpg/2026-10-13-17-10-53_multiquery_pretrain_ctxall-others_s0.0_a0.0_r0.0_best_80137.pt"
JSON="/data1st1/junyi/output/mmllm/AMY_MC/cCRE_cpg/2026-10-13-17-10-53_multiquery_pretrain_ctxall-others_s0.0_a0.0_r0.0_results.json"
TEST_BED="/data1st1/junyi/output/mmllm/AMY_MC/cCRE_cpg/2026-10-13-17-10-53_test_regions.bed"
BED="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"

GPU="${GPU:-5}"
OUT_DIR="output/AMY_MC/cCRE_cpg/xai/test_5hmc"
LOG="logs/xai_shapley_amy_mc_$(date +%Y-%m-%d-%H-%M-%S).log"

for f in "$CKPT" "$JSON" "$TEST_BED" "$BED"; do
  [[ -f "$f" ]] || { echo "ERROR: missing file: $f"; exit 1; }
done
mkdir -p "$OUT_DIR" logs

# refuse to land on a card that is already busy (set GPU=... to override)
used="$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)"
if [[ -z "$used" || "$used" -gt 8000 ]]; then
  echo "ERROR: GPU${GPU} is busy (${used:-unknown} MiB used). Set GPU=<id>."
  exit 1
fi

n_windows="$(wc -l < "$TEST_BED")"
echo "[$(date)] GPU${GPU} | AMY_MC test | ${n_windows} windows | ~41 min"
echo "  log     -> ${LOG}"
echo "  results -> ${OUT_DIR}"

CUDA_VISIBLE_DEVICES="$GPU" python run_xai_modality_shapley.py shapley \
  --checkpoint "$CKPT" \
  --results-json "$JSON" \
  --head 5hmc \
  --split test \
  --region-bed "$BED" \
  --players seq,other,atac,rna,self \
  --absence donor \
  --batch-size 8 \
  --num-workers 4 \
  --out-dir "$OUT_DIR" \
  2>&1 | tee "$LOG"

code="${PIPESTATUS[0]}"
echo "[$(date)] exit=${code}"
exit "$code"
