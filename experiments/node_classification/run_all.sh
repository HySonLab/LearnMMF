#!/bin/bash

DATASETS=("cora" "citeseer")
SPLITS=("MMF1" "MMF2" "MMF3")

for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "Running dataset=$dataset split=$split"
    python main.py \
      --dataset "$dataset" \
      --dim 2 \
      --split "$split" \
      --seed 42 \
      --output-dir "./${dataset}_${split,,}"
  done
done

echo "All runs completed."