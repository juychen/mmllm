#!/bin/bash
# Single-group AMY_MC, cCRE CpG, 2-head model, 100 epochs.
#
# New masking protocol (implemented in run_multiquery_pretraining.py):
#   - select 15% of CpG PAIRS (forward C of each CpG)
#   - hide the pair's C + G, plus +/-50 bp around it, in ALL tracks
#   - loss is computed ONLY at the selected C (not G, not buffer)
#   - modality dropout: ATAC 0.1, RNA 0.1; 5mC/5hmC/seq = 0
#
# Model:
#   query   : 5mC + 5hmC (2 heads)
#   context : ATAC + RNA + DNA, all-others
#   regions : cCRE_cpg.bed
#   test    : cpg_test.bed (chr8/9/X/Y, 7,412 windows)
#
# Usage:
#   GPU=0 ./run_AMY_MC_cCREpair2heads_100ep.sh
#   MASK_NEIGHBORHOOD_BP=100 GPU=0 ./run_AMY_MC_cCREpair2heads_100ep.sh
#   RC=0 GPU=0 ./run_AMY_MC_cCREpair2heads_100ep.sh          # disable RC
#   N_SMOKE=1 GPU=0 ./run_AMY_MC_cCREpair2heads_100ep.sh    # 2,000 windows, 2 epochs

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm || exit 1
export PYTHONUNBUFFERED=1

# ---- paths -----------------------------------------------------------------
M5C="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz"
HM5C="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz"
ATAC="/data1st1/junyi/methdata/atac/AMY_MC_track.bw"
RNA="/data2st1/junyi/output/sn0615/BULK_AMY/AMY_MC.bw"
DMR="/data2st1/junyi/generegion_vM23/cCRE_cpg.bed"
TESTBED="${TESTBED:-output/holdout_testchroms/cpg_test.bed}"

OUT_DIR="${OUT_DIR:-output/AMY_MC/cCRE_cpg}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
GPU="${GPU:-0}"

# ---- settings ---------------------------------------------------------------
NUM_EPOCHS="${NUM_EPOCHS:-100}"
MASK_FRACTION="${MASK_FRACTION:-0.15}"
MASK_NEIGHBORHOOD_BP="${MASK_NEIGHBORHOOD_BP:-50}"
DROP_ATAC="${DROP_ATAC:-0.1}"
DROP_RNA="${DROP_RNA:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLE_SIZES="${SAMPLE_SIZES:-all}"
RC="${RC:-1}"

smoke_args=()
if [[ -n "${N_SMOKE:-}" ]]; then
  NUM_EPOCHS=2
  SAMPLE_SIZES=2000
  BATCH_SIZE=16
  GRAD_ACCUM=1
  NUM_WORKERS=0
  OUT_DIR="/tmp/mq_amy_cpgpair_smoke"
  LOG_DIR="/tmp"
  echo "[smoke] 2,000 windows, 2 epochs, mask_bp=${MASK_NEIGHBORHOOD_BP}"
fi

for f in "$M5C" "$HM5C" "$ATAC" "$RNA" "$DMR" "$TESTBED"; do
  [[ -s "$f" ]] || { echo "[error] missing or empty: $f" >&2; exit 1; }
done
mkdir -p "$OUT_DIR" "$LOG_DIR"

ts="$(date +%Y-%m-%d-%H-%M-%S)"
run_label="multiquery_2heads_cpgpair_bp${MASK_NEIGHBORHOOD_BP}_lossc_drop${DROP_ATAC}_${NUM_EPOCHS}ep"
log_file="${LOG_DIR}/${ts}_${run_label}.out"

rc_args=()
if [[ "$RC" == "1" ]]; then
  rc_args+=(--augment-reverse-complement)
fi

echo "[$(date)] AMY_MC | 2 heads | cCRE CpG | ${NUM_EPOCHS} epochs | mask=CpG-pair hide+/-${MASK_NEIGHBORHOOD_BP}bp, loss=C-only" | tee -a "$log_file"
echo "  drop: atac=${DROP_ATAC} rna=${DROP_RNA} | batch=${BATCH_SIZE} accum=${GRAD_ACCUM} | GPU=${GPU} | RC=${RC}" | tee -a "$log_file"

CUDA_VISIBLE_DEVICES="$GPU" python run_multiquery_pretraining.py \
  --sample-sizes "$SAMPLE_SIZES" \
  --dmr-csv "$DMR" \
  --test-bed "$TESTBED" \
  --target-length 16384 \
  --num-blocks 2 \
  --hidden-dim 64 \
  --fusion-type cross_hyena \
  --query-tracks 5mc 5hmc \
  --context-mode all-others \
  --mask-fraction "$MASK_FRACTION" \
  --mask-neighborhood-bp "$MASK_NEIGHBORHOOD_BP" \
  --drop-p-5mc 0.0 --drop-p-5hmc 0.0 \
  --drop-p-atac "$DROP_ATAC" --drop-p-rna "$DROP_RNA" --drop-p-seq 0.0 \
  "${rc_args[@]}" \
  --scheduler cosine \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --num-workers "$NUM_WORKERS" \
  --scheduler-patience 15 \
  --patience 15 \
  --amp \
  --timestamp "$ts" \
  --scheduler-min-lr 1e-5 \
  --m5c-bedgraph "$M5C" \
  --hm5c-bedgraph "$HM5C" \
  --atac-bw "$ATAC" \
  --rna-coverage-bw "$RNA" \
  --output-csv "${OUT_DIR}/${ts}_${run_label}_results.csv" \
  --output-json "${OUT_DIR}/${ts}_${run_label}_results.json" \
  --best-checkpoint-path "${OUT_DIR}/${ts}_${run_label}_best_{sample_size}.pt" \
  --last-checkpoint-path "${OUT_DIR}/${ts}_${run_label}_last_{sample_size}.pt" \
  2>&1 | tee -a "$log_file"

code=${PIPESTATUS[0]}
if [[ $code -eq 0 ]]; then
  echo "[$(date)] finished" | tee -a "$log_file"
else
  echo "[$(date)] FAILED (exit $code)" | tee -a "$log_file"
fi
exit "$code"
