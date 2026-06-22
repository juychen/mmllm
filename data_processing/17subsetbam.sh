#!/bin/bash
cd /data2st2/junyi/output/atac1112/tobiasbam
for folder in /data2st2/junyi/output/atac1112/tobiasbam/*; do
  echo $folder
  sample_name=$(basename $folder)
  echo "Processing sample: $sample_name"
  for file in $folder/*; do
    echo $file
    if [[ $file == *.csv ]]; then
      celltype=$(basename $folder)
      echo "Processing celltype: $celltype"
      out_bam="${file%.csv}.bam"
      echo "Output BAM: $out_bam"
      # if out_bam already exists, skip
      if [[ -f $out_bam ]]; then
        echo "Output BAM file $out_bam already exists. Skipping."
        continue
      fi
      # Run the subset-bam command
      #printf "Running subset-bam for %s with cell barcodes from %s\n" "$sample_name" "$file"
      /home/junyichen/subset-bam_linux --bam "/data1st2/hannan_25/data/snATAC_process/snATAC_01_bam/$sample_name/outs/possorted_bam.bam" --cell-barcodes "$file" --out-bam "$out_bam" --cores 32
      # Uncomment the line below if you want to run the command for each file
    #   /home/junyichen/subset-bam_linux --bam /data1st2/hannan_25/data/snATAC_process/snATAC_01_bam/$sample_name/outs/possorted_bam.bam \
    #   --cell-barcodes $file \
    #   --out-bam /data1st2/junyi/output/atac0627/tobiasbam/$celltype/${file%.csv}.bam \
    #   --cores 32
    fi
  done
done

# /home/junyichen/subset-bam_linux --bam /data1st2/hannan_25/data/snATAC_process/snATAC_01_bam/MC25A_PFC/outs/possorted_bam.bam \
# --cell-barcodes /data1st2/junyi/output/atac0627/tobiasbam/MC25A_PFC/PFC_L2-3_IT_Glut_MC.csv \
# --out-bam /data1st2/junyi/output/atac0627/tobiasbam/MC25A_PFC/PFC_L2-3_IT_Glut_MC.bam \
# --cores 32