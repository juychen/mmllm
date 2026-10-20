#!/bin/bash
# Sweep the modality-mask (dropout) rate for the multi-query masked pretraining
# run that produced:
#     logs/run_all_allquery_cpg.sh.2026-10-13-17-10-52.out
#
# Baseline (0%) is that run: 100 epochs, hidden=64, 2 blocks, context_mode=all-others,
# mask_fraction=0.15, 80,137 DMRs.  Here only --drop-p-atac and --drop-p-rna change:
# each config applies the SAME p to ATAC and RNA, while DNA sequence and both query
# streams (5mc/5hmc) stay fully visible.  This matches the earlier dropmod grid
# (`submit_dropout_sweep.sh`), so the two are directly comparable.
#
# Note: this is the MODALITY mask (whole-track [MODALITY-MASK] token).  The
# position mask (--mask-fraction, default 0.15) is a separate knob and is left
# untouched; override it with MASK_FRACTION if needed.
#
# The target script takes positional args:
#   $1 fusion_type  $2 seq_drop_p  $3 atac_drop_p  $4 rna_drop_p
# and writes the dropout values into every filename, so configs never collide:
#   ..._multiquery_pretrain_ctxall-others_s0.0_a0.1_r0.1_best_80137.pt
#
# Usage:
#   ./submit_multiquery_dropout_sweep.sh
#
# Environment:
#   DROPS        atac:rna pairs to sweep   (default "0.1:0.1 0.2:0.2 0.4:0.4")
#   FUSION       passed as $1              (default cross_hyena)
#   SEQ_DROP     passed as $2              (default 0.0 — sequence stays visible)
#   PARALLEL     1 = all configs at once (default), 0 = one at a time
#   GPU_STAGGER  seconds between launches when PARALLEL=1 (default 180)
#   SWEEP_GPUS   optional explicit GPU list, one per config, e.g. "1 2 3"
#                (default: each run auto-selects its own idle GPU)
#   BASELINE_LABEL  config shown as the 0% baseline row (default a0.0_r0.0)
#   GROUPS       result dirs for the summary (default "AMY_MC")
#   DRY_RUN      1 = print the commands without running them (default 0)
#
# Forwarded to the target script (env): TEST_H5AD, TEST_H5AD_POSITIONS,
#   MASK_FRACTION, CONTEXT_MODE, BATCH_SIZE, GRAD_ACCUM, NUM_EPOCHS,
#   NUM_WORKERS, MAX_CONCURRENT, REGIONS, CONDITIONS, OUT_TAG.
#
# Examples:
#   DRY_RUN=1 ./submit_multiquery_dropout_sweep.sh
#   SWEEP_GPUS="1 2 3" ./submit_multiquery_dropout_sweep.sh
#   DROPS="0.1:0.1 0.4:0.4" PARALLEL=0 ./submit_multiquery_dropout_sweep.sh
set -uo pipefail

cd /home/junyichen/code/mmllm/ || exit 1

TARGET="./run_all_allquery_cpg.sh"
LOG_DIR="${LOG_DIR:-/home/junyichen/logs}"

DROPS="${DROPS:-0.1:0.1 0.2:0.2 0.4:0.4}"
FUSION="${FUSION:-cross_hyena}"
SEQ_DROP="${SEQ_DROP:-0.0}"
PARALLEL="${PARALLEL:-1}"
GPU_STAGGER="${GPU_STAGGER:-180}"
SWEEP_GPUS="${SWEEP_GPUS:-}"
BASELINE_LABEL="${BASELINE_LABEL:-a0.0_r0.0}"
GROUPS="${GROUPS:-AMY_MC}"
DRY_RUN="${DRY_RUN:-0}"
CONTEXT_MODE="${CONTEXT_MODE:-all-others}"

if [[ ! -x "$TARGET" ]]; then
  echo "Target script not found or not executable: $TARGET (run from /home/junyichen/code/mmllm)" >&2
  exit 1
fi

# ---- validate ---------------------------------------------------------------
in_range() { awk -v p="$1" 'BEGIN{exit !(p >= 0 && p <= 1)}'; }

pairs=()
for pair in $DROPS; do
  atac="${pair%%:*}"
  rna="${pair##*:}"
  if [[ "$pair" != *:* || -z "$atac" || -z "$rna" ]]; then
    echo "Bad DROPS entry '$pair' — expected 'atac:rna', e.g. 0.1:0.1" >&2
    exit 1
  fi
  if ! in_range "$atac" || ! in_range "$rna"; then
    echo "DROPS entry '$pair' out of range [0,1]" >&2
    exit 1
  fi
  pairs+=("$atac:$rna")
done

if [ "${#pairs[@]}" -eq 0 ]; then
  echo "DROPS is empty — nothing to submit" >&2
  exit 1
fi
if ! in_range "$SEQ_DROP"; then
  echo "SEQ_DROP out of range [0,1]: $SEQ_DROP" >&2
  exit 1
fi

sweep_gpus=()
if [[ -n "$SWEEP_GPUS" ]]; then
  read -r -a sweep_gpus <<< "$SWEEP_GPUS"
fi
if [ "${#sweep_gpus[@]}" -gt 0 ] && [ "${#sweep_gpus[@]}" -lt "${#pairs[@]}" ]; then
  echo "SWEEP_GPUS has ${#sweep_gpus[@]} entries but ${#pairs[@]} configs — pad it or drop it" >&2
  exit 1
fi
if [ "$PARALLEL" = "1" ] && [ "${#sweep_gpus[@]}" -eq 0 ] && [ "${#pairs[@]}" -gt 1 ]; then
  echo "note: no SWEEP_GPUS set — each config auto-selects an idle GPU; the ${GPU_STAGGER}s"
  echo "      stagger keeps them from racing onto the same card."
fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "[$(date)] Multi-query modality-mask sweep"
echo "  target        : $TARGET"
echo "  fusion        : $FUSION   (arg \$1)"
echo "  seq dropout   : $SEQ_DROP (arg \$2; 0 = sequence always visible)"
echo "  context mode  : $CONTEXT_MODE"
echo "  mask fraction : ${MASK_FRACTION:-0.15} (position mask, unchanged)"
echo "  configs       : ${#pairs[@]}  ->  ${pairs[*]}"
echo "  mode          : $([ "$PARALLEL" = "1" ] && echo "parallel (stagger ${GPU_STAGGER}s)" || echo sequential)"
echo "  baseline      : $BASELINE_LABEL (existing run, not re-launched)"
echo "  results land  : output/AMY_MC/cCRE_cpg/"
echo "  logs land     : $LOG_DIR/run_all_allquery_cpg.sh.<ts>_a<atac>_r<rna>.out"
[ "$DRY_RUN" = "1" ] && echo "  DRY RUN       : commands will be printed, not executed"
echo "============================================================"

# ---- submit -----------------------------------------------------------------
labels=()
logs=()
i=0
for pair in "${pairs[@]}"; do
  atac="${pair%%:*}"
  rna="${pair##*:}"
  ts="$(date "+%Y-%m-%d-%H-%M-%S")"
  log_file="${LOG_DIR}/run_all_allquery_cpg.sh.${ts}_a${atac}_r${rna}.out"
  labels+=("a${atac}_r${rna}")
  logs+=("$log_file")

  cmd=(env)
  if [ "${#sweep_gpus[@]}" -gt 0 ]; then
    cmd+=("GPUS=${sweep_gpus[$((i % ${#sweep_gpus[@]}))]}")
  fi
  cmd+=("CONTEXT_MODE=${CONTEXT_MODE}"
        nohup "$TARGET" "$FUSION" "$SEQ_DROP" "$atac" "$rna")

  printf '[%s] submit  atac=%s rna=%s  -> %s\n' "$(date '+%H:%M:%S')" "$atac" "$rna" "$log_file"

  if [ "$DRY_RUN" = "1" ]; then
    printf '           %s > %s 2>&1 &\n' "${cmd[*]}" "$log_file"
    i=$((i + 1))
    continue
  fi

  "${cmd[@]}" > "$log_file" 2>&1 &
  pid=$!
  printf '           pid=%s\n' "$pid"

  if [ "$PARALLEL" != "1" ]; then
    wait "$pid" || echo "           WARNING: this config exited non-zero (see $log_file)"
  elif [ "$pair" != "${pairs[-1]}" ]; then
    sleep "$GPU_STAGGER"
  fi
  i=$((i + 1))
done

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "DRY RUN done — nothing was executed."
  exit 0
fi

if [ "$PARALLEL" = "1" ]; then
  echo
  echo "[$(date)] Waiting for all ${#pairs[@]} submissions to finish..."
  wait
fi

# ---- summary ----------------------------------------------------------------
echo
echo "============================================================"
echo "[$(date)] Sweep finished — val (best epoch) / test metrics"
echo "============================================================"
PY=/home/junyichen/anaconda3/envs/evo2/bin/python
"$PY" - "$SEQ_DROP" "$CONTEXT_MODE" "$BASELINE_LABEL" "$GROUPS" "${labels[@]}" <<'PY'
import json, sys
from pathlib import Path

args = sys.argv[1:]
seq_drop, ctx_mode, baseline, groups = args[0], args[1], args[2], args[3]
labels = args[4:]
root = Path("/data1st1/junyi/output/mmllm")

def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"

def find(group, label):
    out_dir = root / group / "cCRE_cpg"
    hits = sorted(out_dir.glob(f"*_multiquery_pretrain_ctx{ctx_mode}_s{seq_drop}_{label}_results.json"))
    if not hits:
        return None
    with open(hits[-1]) as fh:
        data = json.load(fh)
    records = data.get("results") or []
    return records[-1] if records else None

rows = []
for group in groups.split():
    for label in [baseline] + labels:
        rec = find(group, label)
        rows.append((group, label, rec))

header = (f"{'group':<10} {'atac:rna':<10} {'val_5mc':>8} {'val_5hmc':>9} "
          f"{'test_5mc':>9} {'test_5hmc':>10} {'best_ep':>8} {'val_regions':>12}")
print(header)
print("-" * len(header))
for group, label, rec in rows:
    atac, rna = label[1:].split("_r")
    pretty = f"{atac}:{rna}" + (" *" if label == baseline else "")
    if rec is None:
        print(f"{group:<10} {pretty:<10} {'-':>8} {'-':>9} {'(no results json yet)':>20}")
        continue
    vr = rec.get("best_val_r2") or {}
    tr = rec.get("test_r2") or {}
    print(f"{group:<10} {pretty:<10} {fmt(vr.get('5mc')):>8} {fmt(vr.get('5hmc')):>9} "
          f"{fmt(tr.get('5mc')):>9} {fmt(tr.get('5hmc')):>10} {rec.get('best_epoch', '-'):>8} "
          f"{rec.get('val_regions', '-'):>12}")
print()
print("' * ' = baseline (already existing run, not re-launched).  val_regions must be")
print("identical across rows — if not, the split changed and the comparison is invalid.")
print("Full per-run artifacts: output/<group>/cCRE_cpg/<ts>_multiquery_pretrain_ctx" + ctx_mode + "_s" + seq_drop + "_<label>_*")
PY

echo
echo "Per-run logs:"
for log_file in "${logs[@]}"; do
  echo "  $log_file"
done
