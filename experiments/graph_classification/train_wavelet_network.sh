#!/bin/bash

# ============================================================================
# Wavelet Neural Network Trainer
# Usage: ./train_wavelet_network.sh [dataset] [method]
# Examples:
#   ./train_wavelet_network.sh MUTAG baseline
#   ./train_wavelet_network.sh PTC random
#   ./train_wavelet_network.sh DD k_neighbours
#   ./train_wavelet_network.sh NCI1 evolutionary_algorithm
# ============================================================================

# === Parse command line arguments ===
DATASET=$1
METHOD=$2

# === Validate arguments ===
if [ -z "$DATASET" ]; then
    echo "Error: Dataset not specified."
    echo "Usage: ./train_wavelet_network.sh [dataset] [method]"
    echo ""
    echo "Available datasets: MUTAG, PTC, DD, NCI1"
    exit 1
fi

if [ -z "$METHOD" ]; then
    echo "Error: Method not specified."
    echo "Usage: ./train_wavelet_network.sh [dataset] [method]"
    echo ""
    echo "Available methods: baseline, random, k_neighbours, evolutionary_algorithm, directed_evolution"
    exit 1
fi

# === Configuration ===
PROGRAM="train_wavelet_network"
DATA_FOLDER="../../data/"
CONDA_ENV="LearnMMF"

# === Activate conda environment ===
echo "Activating conda environment: $CONDA_ENV..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    echo "Please ensure conda is initialized and the environment exists."
    echo "You can create it with: conda create -n LearnMMF python=3.8"
    exit 1
fi
echo "Conda environment '$CONDA_ENV' activated successfully."
echo ""

# === Hyperparameters ===
NUM_EPOCH=256
NUM_LAYERS=6
HIDDEN_DIM=32

# === Paths ===
BASIS_DIR="$METHOD/$DATASET"
OUTPUT_DIR="$PROGRAM/$METHOD/$DATASET"

# === Check if basis files exist ===
if [ ! -d "$BASIS_DIR" ]; then
    echo "Error: Wavelet basis not found for method '$METHOD' and dataset '$DATASET'."
    echo "Expected directory: $BASIS_DIR"
    echo "Please run generate_wavelet_basis.sh first."
    exit 1
fi

# === Create output directories ===
mkdir -p "$OUTPUT_DIR"

# === Basis file paths ===
ADJS="$BASIS_DIR/$DATASET.$METHOD.adjs.pt"
LAPLACIANS="$BASIS_DIR/$DATASET.$METHOD.laplacians.pt"
MOTHER_WAVELETS="$BASIS_DIR/$DATASET.$METHOD.mother_wavelets.pt"
FATHER_WAVELETS="$BASIS_DIR/$DATASET.$METHOD.father_wavelets.pt"

# === Verify all required files exist ===
for file in "$ADJS" "$LAPLACIANS" "$MOTHER_WAVELETS" "$FATHER_WAVELETS"; do
    if [ ! -f "$file" ]; then
        echo "Error: Missing required file: $file"
        exit 1
    fi
done

# === Display configuration summary ===
echo ""
echo "========================================"
echo "Wavelet Neural Network Training"
echo "========================================"
echo "Dataset:      $DATASET"
echo "Method:       $METHOD"
echo "Output Dir:   $OUTPUT_DIR"
echo "----------------------------------------"
echo "Num Epochs:   $NUM_EPOCH"
echo "Num Layers:   $NUM_LAYERS"
echo "Hidden Dim:   $HIDDEN_DIM"
echo "========================================"
echo ""

# === Cross-validation training ===
for SPLIT in {0..9}; do
    NAME="$PROGRAM.dataset.$DATASET.method.$METHOD.split.$SPLIT.num_epoch.$NUM_EPOCH.num_layers.$NUM_LAYERS.hidden_dim.$HIDDEN_DIM"
    echo "Running split $SPLIT ..."
    python "$PROGRAM.py" \
        --dataset="$DATASET" \
        --data_folder="$DATA_FOLDER" \
        --dir="$OUTPUT_DIR" \
        --name="$NAME" \
        --num_epoch="$NUM_EPOCH" \
        --adjs="$ADJS" \
        --laplacians="$LAPLACIANS" \
        --mother_wavelets="$MOTHER_WAVELETS" \
        --father_wavelets="$FATHER_WAVELETS" \
        --split="$SPLIT" \
        --num_layers="$NUM_LAYERS" \
        --hidden_dim="$HIDDEN_DIM"
done

# === Summary of results ===
echo ""
echo "========================================"
echo "Summary of Results"
echo "========================================"
echo ""

# === Extract best accuracies and save to temporary file ===
RESULTS="$OUTPUT_DIR/accuracies.txt"
[ -f "$RESULTS" ] && rm "$RESULTS"

echo "Extracting best accuracies from logs..."
for SPLIT in {0..9}; do
    NAME="$PROGRAM.dataset.$DATASET.method.$METHOD.split.$SPLIT.num_epoch.$NUM_EPOCH.num_layers.$NUM_LAYERS.hidden_dim.$HIDDEN_DIM"
    LOGFILE="$OUTPUT_DIR/$NAME.log"
    if [ -f "$LOGFILE" ]; then
        grep "Best accuracy:" "$LOGFILE" | awk '{print $3}' >> "$RESULTS"
    fi
done

# === Display individual results ===
if [ -f "$RESULTS" ]; then
    echo "Individual Results:"
    echo "-------------------"
    SPLIT_NUM=0
    while IFS= read -r accuracy; do
        echo "Split $SPLIT_NUM: $accuracy"
        ((SPLIT_NUM++))
    done < "$RESULTS"
    echo ""
    
    # === Calculate mean and std if possible ===
    if command -v python3 &> /dev/null; then
        echo "Statistical Summary:"
        echo "--------------------"
        python3 -c "
import sys
import statistics

try:
    with open('$RESULTS', 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]
    
    if values:
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f'Mean Accuracy: {mean:.4f}')
        print(f'Std Deviation: {std:.4f}')
        print(f'Min Accuracy:  {min(values):.4f}')
        print(f'Max Accuracy:  {max(values):.4f}')
except Exception as e:
    print(f'Could not calculate statistics: {e}')
"
        echo ""
    fi
fi

# ============================================================================
# End
# ============================================================================
echo ""
echo "========================================"
echo "Execution Complete"
echo "========================================"

# === Deactivate conda environment ===
conda deactivate
