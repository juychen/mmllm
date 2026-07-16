#!/bin/bash
# Run all ATAC ablation experiments sequentially.
#
# Ablations (all train on AMY_MC, predict 5hmC):
#   atac_only   : query=ATAC, context=[]              ← KEY: no 5mC input
#   m5c_only    : query=5mC, context=[]               ← KEY: no ATAC input
#   atac_m5c    : query=ATAC, context=[5mC]
#   m5c_atac    : query=5mC,  context=[ATAC]          ← baseline
#   seq_query   : query=DNA,  context=[ATAC, 5mC]
#   seq_only    : query=DNA,  context=[]
#   all_three   : query=concat(5mC,ATAC), context=[DNA]
#
# Each ablation runs with the same hyperparams so results are directly comparable.
#
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

# ---- config ----
ABLATIONS=(
  "m5c_atac"
  "atac_only"
  "m5c_only"
  "atac_m5c"
  "seq_query"
  "all_three"
)

DATASET_FLAGS=(
  --dmr-csv /data2st1/junyi/generegion_vM23/cCRE_cpg.bed
  --m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz
  --hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz
  --atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw
  --genome-fasta /data2st1/junyi/ref/GRCm38.p6.genome.fa
)

TRAINING_FLAGS=(
  --sample-sizes 5000
  --target-length 16384
  --batch-size 8
  --gradient-accumulation-steps 64
  --mask-mode cpg_forward
  --augment-reverse-complement
  --num-epochs 100
  --scheduler-patience 10
  --patience 10
  --scheduler cosine
  --learning-rate 1e-3
  --weight-decay 1e-5
  --amp
  --gradient-checkpointing
  --lazy
  --hidden-dim 64
  --model-b-blocks 2
  --model-b-fusion cross_hyena
  --output-dir output/atac_ablation
)

mkdir -p output/atac_ablation

summary_file="output/atac_ablation/_summary.tsv"
echo -e "ablation\tquery\tcontext\tbest_val_loss\tbest_val_r2\tbest_val_pearson" > "$summary_file"

for ablation in "${ABLATIONS[@]}"; do
  echo ""
  echo "============================================"
  echo "[$(date)] Running ablation: $ablation"
  echo "============================================"

  python run_atac_ablation.py \
    --ablation "$ablation" \
    "${DATASET_FLAGS[@]}" \
    "${TRAINING_FLAGS[@]}"

  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    latest_metrics=$(ls -t "output/atac_ablation/${ablation}/"*.json 2>/dev/null | head -1)
    if [[ -f "$latest_metrics" ]]; then
      python -c "
import json, sys
with open('$latest_metrics') as f:
    m = json.load(f)
print(f'  {m[\"ablation\"]}\\t{m[\"query_modality\"]}\\t{\",\".join(m[\"context_modalities\"])}\\t{m[\"best_val_loss\"]:.4f}\\t{m[\"best_val_r2\"]:.4f}\\t{m[\"best_val_pearson\"]:.4f}', flush=True)
" | tee -a "$summary_file"
    fi
  else
    echo "  ✗ $ablation FAILED"
  fi
done

echo ""
echo "============================================"
echo "[$(date)] All ablations complete."
echo ""
echo "Summary:"
column -t -s $'\t' "$summary_file"
echo "============================================"