#!/bin/bash
# Submit `run_all_m5c_query_26_dropmod.sh` several times, once per
# (atac_drop_p, rna_drop_p) pair, so the effect of the modality-dropout rate can
# be compared on the AMY groups.
#
# The target script takes positional args:
#   $1 fusion_type  $2 seq_drop_p  $3 atac_drop_p  $4 rna_drop_p
# so each sweep is a separate submission with its own dropout values. Results do
# not collide: the target writes into output/<GROUP>/cCRE_cpg/ with the dropout
# values baked into every filename (`..._dropmod_s<seq>_a<atac>_r<rna>_...`).
#
# Note the `submit` alias cannot be used here — it only runs `./$1` and cannot
# forward positional arguments.
#
# Usage:
#   ./submit_dropout_sweep.sh
#
# Environment:
#   DROPS        atac:rna pairs to sweep   (default "0.0:0.0 0.1:0.1 0.2:0.2 0.4:0.4")
#   FUSION       passed as $1              (default cross_hyena)
#   SEQ_DROP     passed as $2              (default 0.0)
#   BATCH_SIZE   forwarded to the target   (default 8)
#   GRAD_ACCUM   forwarded to the target   (default 32; 8x32 == the old 4x64)
#   PARALLEL     0 = one config at a time  (default), 1 = submit all at once
#   GPU_STAGGER  seconds between launches when PARALLEL=1 (default 90)
#   DRY_RUN      1 = print the commands without running them (default 0)
#
# Examples:
#   ./submit_dropout_sweep.sh                                   # full sweep, sequential
#   DROPS="0.1:0.1 0.4:0.4" ./submit_dropout_sweep.sh           # just the two rates asked for
#   DRY_RUN=1 ./submit_dropout_sweep.sh                         # show what would run
#   PARALLEL=1 GPU_STAGGER=120 ./submit_dropout_sweep.sh        # all configs at once, staggered
set -uo pipefail

cd /home/junyichen/code/mmllm/ || exit 1

TARGET="./run_all_m5c_query_26_dropmod.sh"
LOG_DIR="/home/junyichen/logs"

DROPS="${DROPS:-0.0:0.0 0.1:0.1 0.2:0.2 0.4:0.4}"
FUSION="${FUSION:-cross_hyena}"
SEQ_DROP="${SEQ_DROP:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
PARALLEL="${PARALLEL:-0}"
GPU_STAGGER="${GPU_STAGGER:-90}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$TARGET" ]]; then
  echo "Target script not found: $TARGET (run from /home/junyichen/code/mmllm)" >&2
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

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "[$(date)] Modality-dropout sweep on the AMY groups"
echo "  target       : $TARGET"
echo "  fusion       : $FUSION   (arg \$1)"
echo "  seq dropout  : $SEQ_DROP (arg \$2)"
echo "  batch size   : $BATCH_SIZE (micro) x $GRAD_ACCUM (accum) = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "  configs      : ${#pairs[@]}  ->  ${pairs[*]}"
echo "  mode         : $([ "$PARALLEL" = "1" ] && echo "parallel (stagger ${GPU_STAGGER}s)" || echo sequential)"
echo "  results land : output/AMY_MC/cCRE_cpg/ and output/AMY_MW/cCRE_cpg/"
echo "  logs land    : $LOG_DIR/run_all_m5c_query_26_dropmod.sh.<ts>_a<atac>_r<rna>.out"
[ "$DRY_RUN" = "1" ] && echo "  DRY RUN      : commands will be printed, not executed"
echo "============================================================"

# ---- submit -----------------------------------------------------------------
labels=()
logs=()
for pair in "${pairs[@]}"; do
  atac="${pair%%:*}"
  rna="${pair##*:}"
  ts="$(date "+%Y-%m-%d-%H-%M-%S")"
  log_file="${LOG_DIR}/run_all_m5c_query_26_dropmod.sh.${ts}_a${atac}_r${rna}.out"
  labels+=("a${atac}_r${rna}")
  logs+=("$log_file")

  cmd=(env "BATCH_SIZE=${BATCH_SIZE}" "GRAD_ACCUM=${GRAD_ACCUM}"
       nohup "$TARGET" "$FUSION" "$SEQ_DROP" "$atac" "$rna")
  printf '[%s] submit  atac=%s rna=%s  -> %s\n' "$(date '+%H:%M:%S')" "$atac" "$rna" "$log_file"

  if [ "$DRY_RUN" = "1" ]; then
    printf '           %s\n' "${cmd[*]} > ${log_file} 2>&1 &"
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
echo "[$(date)] Sweep finished — best-epoch validation metrics"
echo "============================================================"
PY=/home/junyichen/anaconda3/envs/evo2/bin/python
"$PY" - "$SEQ_DROP" "${labels[@]}" <<'PY'
import json, sys
from pathlib import Path

seq_drop, labels = sys.argv[1], sys.argv[2:]
root = Path("/data1st1/junyi/output/mmllm")
groups = ("AMY_MC", "AMY_MW")

def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"

rows = []
for group in groups:
    out_dir = root / group / "cCRE_cpg"
    for label in labels:
        # filename carries the dropout values: ..._dropmod_s<seq>_a<atac>_r<rna>_results.json
        hits = sorted(out_dir.glob(f"*_dropmod_s{seq_drop}_{label}_results.json"))
        if not hits:
            rows.append((group, label, None))
            continue
        with open(hits[-1]) as fh:
            data = json.load(fh)
        records = data.get("results") or []
        rows.append((group, label, records[-1] if records else None))

header = (f"{'group':<8} {'atac:rna':<10} {'best_val_loss':>13} {'best_val_r2':>11} "
          f"{'best_val_pearsonr':>17} {'best_epoch':>10} {'val_regions':>11}")
print(header)
print("-" * len(header))
for group, label, rec in rows:
    atac, rna = label[1:].split("_r")
    pretty = f"{atac}:{rna}"
    if rec is None:
        print(f"{group:<8} {pretty:<10} {'-':>13} {'-':>11} {'(no results json yet)':>17} {'-':>10} {'-':>11}")
        continue
    print(f"{group:<8} {pretty:<10} {fmt(rec.get('best_val_loss')):>13} "
          f"{fmt(rec.get('best_val_r2')):>11} {fmt(rec.get('best_val_pearsonr')):>17} "
          f"{rec.get('best_epoch', '-'):>10} {rec.get('val_regions', '-'):>11}")
print()
print("val_regions should be identical across configs — if it is not, the dropout rate")
print("changed the train/val split and the comparison is not apples-to-apples.")
PY

echo
echo "Per-run logs:"
for log_file in "${logs[@]}"; do
  echo "  $log_file"
done
