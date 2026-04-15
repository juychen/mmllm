#!/bin/bash
set -euo pipefail

source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate snapatac2

INPUT_DIR=/data1st2/hannan_25/data/Nanopore_processV1/nanopore_06_distribution
OUTPUT_DIR=/data2st1/junyi/output/llm0401/processed_meth

mkdir -p "$OUTPUT_DIR"
cd "$INPUT_DIR"
shopt -s nullglob

for condition in MC MW; do
    for region in HIP AMY PFC; do
        for context in CH CG; do
            for suffix in m h; do
                pattern="${condition}*-${region}.6mA_5mC5hmC.${context}.pileup.${suffix}.bigwig"
                files=($pattern)

                if [ ${#files[@]} -eq 0 ]; then
                    echo "跳过：没有找到匹配 ${pattern} 的文件。"
                    continue
                fi

                output_file="${OUTPUT_DIR}/${condition}_${region}.${context}.${suffix}.bedGraph"

                if [ -f "$output_file" ]; then
                    echo "跳过：${output_file} 已存在。"
                    continue
                fi

                echo "Merging ${pattern} -> ${output_file}"
                if ! wiggletools write_bg "$output_file" mean "${files[@]}"; then
                    echo "跳过：${pattern} 处理失败。"
                    rm -f "$output_file"
                    continue
                fi
            done
        done
    done
done

for bg in "$OUTPUT_DIR"/*.bedGraph; do
    [ -f "$bg" ] || continue
    if [ -f "${bg}.gz" ]; then
        echo "跳过：${bg}.gz 已存在。"
        continue
    fi
    echo "Compressing and indexing ${bg}..."
    bgzip "$bg" && tabix -p bed "${bg}.gz"
done 