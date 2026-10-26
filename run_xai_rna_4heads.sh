#!/bin/bash
# XAI on the 4-head 30-epoch checkpoint, with RNA as the target head.
#
# Players: DNA sequence + 5mC + 5hmC + ATAC (4 players -> 16 coalitions).
# The RNA self channel is NOT a player, so nothing is attributed to the RNA
# neighbourhood; the other modalities' contribution to the RNA head is what
# this run measures.  NOTE: the RNA track itself is still visible to the model
# at unmasked positions (it is simply not a player). If you want the RNA
# neighbourhood fully removed for every coalition, a fixed-absence switch is
# needed (not available in the current script).
#
# Output: output/AMY_MC/cCRE_cpg/xai/4heads30_rna_noself/
# Log:    /home/junyichen/logs/xai_4heads30_rna_noself_<timestamp>.out

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1
export PYTHONUNBUFFERED=1

CKPT="output/AMY_MC/cCRE_cpg/2026-10-21-17-19-37_multiquery_4heads_ctxall-others_best_80137.pt"
JSON="output/AMY_MC/cCRE_cpg/2026-10-21-17-19-37_multiquery_4heads_ctxall-others_results.json"
OUT_DIR="output/AMY_MC/cCRE_cpg/xai/4heads30_rna_noself"
GPU="${GPU:-0}"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="/home/junyichen/logs/xai_4heads30_rna_noself_${TS}.out"

mkdir -p "$OUT_DIR"
echo "[$(date)] RNA head XAI, 4 players (no self) | GPU${GPU}" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" python run_xai_modality_shapley.py shapley \
  --checkpoint "$CKPT" \
  --results-json "$JSON" \
  --head rna \
  --split test \
  --players seq,5mc,5hmc,atac \
  --absence donor \
  --batch-size 8 \
  --num-workers 4 \
  --out-dir "$OUT_DIR" \
  2>&1 | tee -a "$LOG"

code=${PIPESTATUS[0]}
echo "[$(date)] exit=${code}" | tee -a "$LOG"
exit "$code"
