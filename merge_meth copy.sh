#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate snapatac2
# 用法: ./merge_meth.sh output_name.bw input1.bw input2.bw input3.bw
OUTPUT=$1
shift
INPUTS=$@

# 1. 下载 mm10 chrom.sizes (如果不存在)
if [ ! -f mm10.chrom.sizes ]; then
    echo "Downloading mm10.chrom.sizes..."
    curl -L http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes -o mm10.chrom.sizes
fi

echo "Merging [ $INPUTS ] into $OUTPUT..."

# 2. 管道流：
# wiggletools write_bg - : 将平均值以 bedGraph 格式输出到 stdout
# awk : 确保数值在 [0, 1] 之间（防止浮点溢出导致转换失败）
# bedGraphToBigWig : 从 stdin 读取并生成最终的 BigWig
wiggletools write_bg $OUTPUT.bedGraph mean $INPUTS