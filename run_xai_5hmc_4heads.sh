#!/bin/bash
# XAI on the 4-head 30-epoch checkpoint, 5hmC head.
# Two player sets run in parallel on GPU0 / GPU1:
#   5-player : seq + 5mc + ATAC + RNA + 5hmC self
#   4-player : seq + 5mc + ATAC + RNA          (no self)
#
# Logs: /home/junyichen/logs/xai_4heads30_5hmc_{5p,4p}_<timestamp>.out
# Outputs: output/AMY_MC/cCRE_cpg/xai/4heads30_5hmc{,_noself}/

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1
export PYTHONUNBUFFERED=1

CKPT="output/AMY_MC/cCRE_cpg/2026-10-21-17-19-37_multiquery_4heads_ctxall-others_best_80137.pt"
JSON="output/AMY_MC/cCRE_cpg/2026-10-21-17-19-37_multiquery_4heads_ctxall-others_results.json"
OUT_ROOT="output/AMY_MC/cCRE_cpg/xai"
TS="$(date +%Y-%m-%d-%H-%M-%S)"

mkdir -p "$OUT_ROOT/4heads30_5hmc" "$OUT_ROOT/4heads30_5hmc_noself"

CUDA_VISIBLE_DEVICES=0 python run_xai_modality_shapley.py shapley \
  --checkpoint "$CKPT" --results-json "$JSON" \
  --head 5hmc --split test \
  --players seq,other,atac,rna,self \
  --absence donor --batch-size 8 --num-workers 4 --tag 5p \
  --out-dir "$OUT_ROOT/4heads30_5hmc" \
  > "/home/junyichen/logs/xai_4heads30_5hmc_5p_${TS}.out" 2>&1 &
PID_5P=$!

CUDA_VISIBLE_DEVICES=1 python run_xai_modality_shapley.py shapley \
  --checkpoint "$CKPT" --results-json "$JSON" \
  --head 5hmc --split test \
  --players seq,other,atac,rna \
  --absence donor --batch-size 8 --num-workers 4 --tag 4p \
  --out-dir "$OUT_ROOT/4heads30_5hmc_noself" \
  > "/home/junyichen/logs/xai_4heads30_5hmc_4p_${TS}.out" 2>&1 &
PID_4P=$!

echo "[$(date)] launched 5p=$PID_5P 4p=$PID_4P ts=$TS"
wait "$PID_5P"; CODE_5P=$?
wait "$PID_4P"; CODE_4P=$?
echo "[$(date)] 5-player exit=$CODE_5P | 4-player exit=$CODE_4P"
[[ $CODE_5P -eq 0 && $CODE_4P -eq 0 ]]
