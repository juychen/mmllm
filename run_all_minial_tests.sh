#!/bin/bash
source /home/junyichen/anaconda3/etc/profile.d/conda.sh
conda activate evo2
cd /home/junyichen/code/mmllm/ || exit 1

regions=("${@}")
conditions=(MC MW)

if [[ ${#regions[@]} -eq 0 ]]; then
  regions=(AMY HIP PFC)
fi

for region in "${regions[@]}"; do
  if [[ "$region" != "AMY" && "$region" != "HIP" && "$region" != "PFC" ]]; then
    echo "Unsupported region: $region"
    echo "Allowed values: AMY HIP PFC"
    exit 1
  fi
done

for condition in "${conditions[@]}"; do
  for region in "${regions[@]}"; do
    echo "Running region=$region condition=$condition"
    ./run_minial_test.sh "$region" "$condition"
  done
done