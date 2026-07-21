#!/bin/bash
# Merge per-context (CG, CH) bedGraph files into a single whole-cytosine (.C) track.
#
# Usage:
#   bash run_merge_cg_ch_to_c.sh
#
# Edit INPUT_DIR / OUTPUT_DIR below if your files live elsewhere.
#
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

export PYTHONUNBUFFERED=1

# ---- config ----
INPUT_DIR="/data2st1/junyi/output/llm0401/processed_meth"
OUTPUT_DIR="$INPUT_DIR"   # write into same directory by default

# ---- sanity check: confirm input files exist ----
echo "Checking input directory: $INPUT_DIR"
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: input directory not found: $INPUT_DIR"
  exit 1
fi

# Show a sample of files to confirm naming convention
echo ""
echo "Sample of available bedGraph files:"
ls "$INPUT_DIR" | grep -E '\.(CG|CH)\.(m|h)\.bedGraph' | head -8
echo ""

# ---- run merge (5hmC only: CG.h + CH.h → .C.h) ----
echo "============================================"
echo "[$(date)] Merging CG + CH 5hmC bedGraphs → .C.h tracks"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

python merge_cg_ch_to_c.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --modalities h \
  --skip-existing

exit_code=$?

echo ""
echo "============================================"
if [ $exit_code -eq 0 ]; then
  echo "[$(date)] Merge finished successfully!"
  echo "Generated .C.h.bedGraph.gz files:"
  ls -lh "$OUTPUT_DIR"/*.C.h.bedGraph.gz 2>/dev/null | awk '{printf "  %-12s  %s\n", $5, $9}'
else
  echo "[$(date)] Merge FAILED with exit code $exit_code"
fi
echo "============================================"

exit $exit_code