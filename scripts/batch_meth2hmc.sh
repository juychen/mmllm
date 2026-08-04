#!/usr/bin/env bash
# Run meth2hmc.py on every (sample) where both *_BS_d2_*.meth.txt.gz and
# *_oxBS_d2_*.meth.txt.gz exist under METH_DIR. Skips if output is up-to-date.
#
# Outputs go into OUT_DIR (default: same place the BS bw files live).
# Optional flag --bb-posterior switches the mode (slow: ~20 min/sample).
set -euo pipefail

METH_DIR=${1:-/data1st1/junyi/methdata/GSE214845/meth}
OUT_DIR=${2:-/data1st1/junyi/methdata/GSE214845}
PY=${PY:-/home/junyichen/anaconda3/bin/python3}
SCRIPT=$(dirname "$(readlink -f "$0")")/meth2hmc.py
MIN_COV=${MIN_COV:-5}
HEADER_BW="${HEADER_BW:-$OUT_DIR/GSE214845_WT72_D2_BS.bw}"
BB_FLAG=""
[[ "${BB:-0}" == "1" ]] && BB_FLAG="--bb-posterior"

# WT72 already done — pick the first sample we find to use as header reference
if [[ ! -e "$HEADER_BW" ]]; then
    echo "[err] header bw not found: $HEADER_BW" >&2; exit 1
fi

for bs in "$METH_DIR"/GSM*_BS_d2_*.meth.txt.gz; do
    [[ -e "$bs" ]] || continue
    base=${bs##*/}                       # e.g. GSM6616456_BS_d2_WT72.deduplicated.sorted.meth.txt.gz
    sample=${base#GSM*_BS_d2_}           # WT72.deduplicated.sorted.meth.txt.gz
    sample=${sample%.deduplicated.sorted.meth.txt.gz}  # WT72
    ox="$METH_DIR/GSM*_oxBS_d2_${sample}.deduplicated.sorted.meth.txt.gz"
    ox=$(ls $ox 2>/dev/null | head -1)
    [[ -n "$ox" && -e "$ox" ]] || { echo "[skip] no oxBS for $sample"; continue; }

    out_bw="$OUT_DIR/GSE214845_${sample}_D2_5hmC.bw"
    out_bg="$OUT_DIR/GSE214845_${sample}_D2_5hmC.bedGraph.gz"

    if [[ -e "$out_bw" ]] && [[ "$out_bw" -nt "$bs" ]] && [[ "$out_bw" -nt "$ox" ]]; then
        echo "[ok-already] $(basename "$out_bw")"
        continue
    fi

    echo "[run] $sample ($(basename "$bs")) -> $(basename "$out_bw")"
    "$PY" "$SCRIPT" \
        --bs   "$bs" --oxbs "$ox" \
        --out  "$out_bw" \
        --out-bedgraph "$out_bg" \
        --header-bw "$HEADER_BW" \
        --min-cov "$MIN_COV" $BB_FLAG
done
echo "[done] all samples"
