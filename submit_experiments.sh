#!/bin/bash
# =============================================================================
# submit_experiments.sh
#
# Submit an experiment launcher (`run_all_*.sh`) as a detached job and mirror ALL
# of its stdout/stderr into a timestamped wrapper log under /home/junyichen/logs
# -- the same convention as the existing scheduler-style logs, e.g.
#
#   /home/junyichen/logs/run_all_masked_multimodal_cpg.sh.2026-10-12-20-48-12.out
#   /home/junyichen/logs/run_all_multiquery_cpg.sh.2026-10-13-15-30-00.out
#
# Each underlying run still writes its own per-run log next to its outputs
# (`output/<REGION>_<COND>/cCRE_cpg/<timestamp>_<label>.log`); this wrapper log is
# the top-level record of the whole submission (banner, per-job lines, summary).
# =============================================================================
set -u

WORKDIR="${WORKDIR:-/home/junyichen/code/mmllm}"
LOG_DIR="${LOG_DIR:-/home/junyichen/logs}"
DEFAULT_TARGET="run_all_multiquery_cpg.sh"

usage() {
  cat <<'EOF'
Usage:
  ./submit_experiments.sh [launcher.sh] [launcher args...]
  ./submit_experiments.sh --list
  ./submit_experiments.sh --help

Examples:
  # default: all 6 groups, multi-query pretraining, shared context (detached)
  ./submit_experiments.sh

  # AMY_MC only, multi-query with all-others context, on GPU 4, stay attached
  REGIONS=AMY CONDITIONS=MC CONTEXT_MODE=all-others GPUS=4 FOREGROUND=1 \
    ./submit_experiments.sh run_all_multiquery_cpg.sh cross_hyena 0.0 0.0 0.0

  # the older single-query masked-multimodal launcher, same logging
  ./submit_experiments.sh run_all_masked_multimodal_cpg.sh cross_hyena 0.0 0.0 0.0

  # ~3-minute smoke test on a spare GPU, logged the same way
  N_SMOKE=1 REGIONS=AMY CONDITIONS=MC GPUS=6 FOREGROUND=1 ./submit_experiments.sh

Env (passed through to the launcher unchanged):
  LOG_DIR      where wrapper logs go           (default /home/junyichen/logs)
  FOREGROUND   1 = stay attached; default is detached (nohup + disown)
  REGIONS CONDITIONS GPUS MAX_CONCURRENT CONTEXT_MODE MASK_FRACTION
  BATCH_SIZE GRAD_ACCUM NUM_WORKERS NUM_EPOCHS OUT_TAG TEST_BED N_SMOKE ...

Wrapper log path:  <LOG_DIR>/<launcher basename>.<YYYY-MM-DD-HH-MM-SS>.out
EOF
}

case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
  --list|-l)
    echo "Available launchers in $WORKDIR:"
    ( cd "$WORKDIR" 2>/dev/null && ls -1 run_all_*.sh 2>/dev/null | sed 's/^/  /' )
    exit 0 ;;
esac

target="${1:-$DEFAULT_TARGET}"
[[ $# -gt 0 ]] && shift

if [[ ! -f "$WORKDIR/$target" ]]; then
  echo "ERROR: launcher '$target' not found in $WORKDIR" >&2
  echo "       run '$0 --list' to see the available launchers." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
ts="$(date '+%Y-%m-%d-%H-%M-%S')"
log="$LOG_DIR/$(basename "$target").$ts.out"

# --- provenance header, written into the wrapper log -------------------------
{
  echo "=============================================================================="
  echo "[$(date '+%Y年 %m月 %d日 %A %H:%M:%S %Z')] submit_experiments.sh"
  echo "  target   : $target $*"
  echo "  workdir  : $WORKDIR"
  echo "  log      : $log"
  printf '  env      :'
  printed=0
  for v in REGIONS CONDITIONS GPUS MAX_CONCURRENT CONTEXT_MODE MASK_FRACTION \
           BATCH_SIZE GRAD_ACCUM NUM_WORKERS NUM_EPOCHS OUT_TAG TEST_BED N_SMOKE; do
    if [[ -n "${!v:-}" ]]; then printf ' %s=%s' "$v" "${!v}"; printed=1; fi
  done
  [[ "$printed" == "0" ]] && printf ' (none set; launcher defaults)'
  echo
  echo "=============================================================================="
  echo
} > "$log"

cd "$WORKDIR" || { echo "ERROR: cannot cd $WORKDIR" >&2; exit 1; }

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[$(date '+%H:%M:%S')] launching attached: $target $*"
  echo "[$(date '+%H:%M:%S')] wrapper log: $log"
  echo "------------------------------------------------------------------------------"
  bash "$target" "$@" 2>&1 | tee -a "$log"
  rc="${PIPESTATUS[0]}"
  echo "------------------------------------------------------------------------------"
  echo "[$(date '+%H:%M:%S')] finished (exit=$rc) | log: $log"
  exit "$rc"
else
  nohup bash "$target" "$@" >> "$log" 2>&1 &
  pid=$!
  disown 2>/dev/null || true
  echo ""
  echo "============================================================"
  echo "[$(date '+%Y年 %m月 %d日 %A %H:%M:%S %Z')] Submitted"
  echo "  launcher : $target $*"
  echo "  pid      : $pid"
  echo "  log      : $log"
  echo "  follow   : tail -f $log"
  echo "============================================================"
fi
