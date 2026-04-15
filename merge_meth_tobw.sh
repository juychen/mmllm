#!/bin/bash
set -euo pipefail

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate snapatac2

INPUT_DIR=/data2st1/junyi/output/llm0401/processed_meth
OUTPUT_DIR=/data2st1/junyi/output/llm0401/processed_meth
CHROM_SIZES=/home/junyichen/code/mmllm/mm10.chrom.sizes

# Download mm10 chrom.sizes if missing
if [ ! -f "$CHROM_SIZES" ]; then
    echo "Downloading mm10.chrom.sizes..."
    curl -L http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes \
         -o "$CHROM_SIZES"
fi

mkdir -p "$OUTPUT_DIR"

for bg in "$INPUT_DIR"/*.bedGraph; do
    [ -f "$bg" ] || continue
    bw="${bg%.bedGraph}.bw"
    if [ -f "$bw" ]; then
        echo "跳过：${bw} 已存在。"
        continue
    fi

    echo "Converting ${bg} -> ${bw}..."
    bedGraphToBigWig "$bg" "$CHROM_SIZES" "$bw" \
    || { echo "失败：${bg} 转换失败，跳过。"; rm -f "$bw"; }
done

echo "全部完成。"

