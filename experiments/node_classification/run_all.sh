#!/bin/bash

DATASETS=("cora" "citeseer")
SPLITS=("MMF1" "MMF2" "MMF3")

for dataset in "${DATASETS[@]}"; do
    python main.py \
      --dataset "$dataset" \
      --dim 2 \
      --seed 42 \
      --output-dir "./${dataset}_wavelets" \
      --save-wavelets
done

for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "Running dataset=$dataset split=$split"
    python main.py \
      --dataset "$dataset" \
      --dim 2 \
      --split "$split" \
      --seed 42 \
      --output-dir "./${dataset}_${split,,}" \
      --load-wavelets "./${dataset}_wavelets/wavelets"
  done
done

echo "All runs completed."