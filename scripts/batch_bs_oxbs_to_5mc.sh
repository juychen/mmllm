#!/usr/bin/env bash
# Run bw_bs_oxbs_to_5mc.py on every (sample) where both *_D2_BS.bw and
# *_D2_oxBS.bw exist under the given input dir. WT72 is skipped if its 5mC
# already exists.
set -euo pipefail

IN=${1:-/data1st1/junyi/methdata/GSE214845}
PY=${PY:-/home/junyichen/anaconda3/bin/python3}
SCRIPT=$(dirname "$(readlink -f "$0")")/bw_bs_oxbs_to_5mc.py

for bs in "$IN"/GSE214845_*_D2_BS.bw; do
    [[ -e "$bs" ]] || continue
    base=${bs%_D2_BS.bw}
    ox="${base}_D2_oxBS.bw"
    out="${base}_D2_5mC.bw"
    bg="${base}_D2_5mC.bedGraph.gz"
    [[ -e "$ox" ]] || { echo "[skip] no oxBS for $(basename "$base")"; continue; }
    if [[ -e "$out" ]] && [[ "$out" -nt "$bs" ]] && [[ "$out" -nt "$ox" ]]; then
        echo "[ok-already] $(basename "$out")"
        continue
    fi
    echo "[run] $(basename "$bs") + $(basename "$ox") -> $(basename "$out")"
    "$PY" "$SCRIPT" --bs "$bs" --oxbs "$ox" --out "$out" --bedgraph "$bg"
done
echo "[done] all samples"
