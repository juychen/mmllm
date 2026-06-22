#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate tobias
ulimit -n 65534

cd /data2st2/junyi/output/atac1112/tobiasbam
folder1="/data2st2/junyi/output/atac1112/tobiasbam/MW65A_AMY/"
folder2="/data2st2/junyi/output/atac1112/tobiasbam/MW51A_AMY/"
folder3="/data2st2/junyi/output/atac1112/tobiasbam/MW47A_AMY/"

# Define the output folder for merged files
output_dir="/data2st2/junyi/output/atac1112/tobiasbam/MW_AMY"
mkdir -p "$output_dir"  # Create output directory if it doesn't exist
mkdir -p "$output_dir/finish_list"

# Iterate through all .bam files in the first folder
for bam_file in "$folder1"/*.bam; do
    # Extract the base filename (without path)
    filename=$(basename "$bam_file")
    
    # Check if the same filename exists in the other two folders
    if [[ -f "$folder2/$filename" && -f "$folder3/$filename" ]]; then
        echo "Merging $filename..."
        echo $folder1/$filename
        echo $folder2/$filename
        echo $folder3/$filename
        # Merge the three files into one
        # if the finish file is not in the folder
        if [[ ! -f "$output_dir/finish_list/$filename.txt" ]]; then
            samtools merge -f "$output_dir/$filename" \
                "$folder1/$filename" \
                "$folder2/$filename" \
                "$folder3/$filename" --threads 16

            # if samtools finished successfully
            if [[ $? -eq 0 ]]; then
                touch "$output_dir/finish_list/$filename.txt"
            fi
        fi
        # Index the merged file (optional)
        # samtools index "$output_dir/$filename"
        # If finish the write a text to '/data2st2/junyi/output/atac1112/tobiasbam/MW_AMY/finish_list'
    else
        echo "Skipping $filename (not found in all folders)"
    fi
done

echo "Merging complete. Output saved to $output_dir."
