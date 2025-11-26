#!/bin/bash

# ============================================================================
# Parametric Experiment Runner with Timing
# Usage: ./generate_wavelet_basis.sh [dataset] [method]
# Examples:
#   ./generate_wavelet_basis.sh MUTAG baseline
#   ./generate_wavelet_basis.sh PTC evolutionary_algorithm
#   ./generate_wavelet_basis.sh DD directed_evolution
#   ./generate_wavelet_basis.sh NCI1 k_neighbours
# ============================================================================

# === Parse command line arguments ===
DATASET=$1
METHOD=$2

# === Validate arguments ===
if [ -z "$DATASET" ]; then
    echo "Error: Dataset not specified"
    echo ""
    echo "Usage: ./generate_wavelet_basis.sh [dataset] [method]"
    echo ""
    echo "Available datasets:"
    echo "  MUTAG, PTC, DD, NCI1"
    echo ""
    echo "Available methods:"
    echo "  baseline, random, k_neighbours, evolutionary_algorithm, directed_evolution"
    echo ""
    echo "Example: ./generate_wavelet_basis.sh MUTAG baseline"
    exit 1
fi

if [ -z "$METHOD" ]; then
    echo "Error: Method not specified"
    echo ""
    echo "Usage: ./generate_wavelet_basis.sh [dataset] [method]"
    echo ""
    echo "Available methods:"
    echo "  baseline              - Baseline MMF"
    echo "  random                - Random MMF"
    echo "  k_neighbours          - k-Neighbors MMF"
    echo "  evolutionary_algorithm- Evolutionary Algorithm MMF"
    echo "  directed_evolution    - Directed Evolution MMF"
    echo ""
    echo "Example: ./generate_wavelet_basis.sh MUTAG baseline"
    exit 1
fi

# === Configuration ===
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

# === Set dataset-specific parameters ===
case "${DATASET^^}" in
    MUTAG)
        DIM=2
        K=2
        DROP=1
        EPOCHS=1024
        LEARNING_RATE=1e-3
        ;;
    PTC)
        DIM=2
        K=2
        DROP=1
        EPOCHS=1024
        LEARNING_RATE=1e-3
        ;;
    DD)
        DIM=2
        K=2
        DROP=1
        EPOCHS=1024
        LEARNING_RATE=1e-3
        ;;
    NCI1)
        DIM=2
        K=2
        DROP=1
        EPOCHS=1024
        LEARNING_RATE=1e-3
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo ""
        echo "Available datasets: MUTAG, PTC, DD, NCI1"
        exit 1
        ;;
esac

# === Create timing log directory ===
mkdir -p "timing_logs"
TIMING_LOG="timing_logs/timing_summary.txt"

# === Display configuration ===
echo ""
echo "========================================"
echo "Wavelet Basis Generation with Timing"
echo "========================================"
echo "Dataset: $DATASET"
echo "Method:  $METHOD"
echo "========================================"
echo ""

# === Record start time ===
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_SECONDS=$(date +%s)

# === Helper function to setup directories ===
setup_directories() {
    local dir=$1
    mkdir -p "$dir/$DATASET"
}

# === Helper function to log completion ===
log_completion() {
    local method_name=$1
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    END_SECONDS=$(date +%s)
    DURATION=$((END_SECONDS - START_SECONDS))
    
    echo ""
    echo "========================================"
    echo "$method_name completed for $DATASET"
    echo "========================================"
    echo "Start time:  $START_TIME"
    echo "End time:    $END_TIME"
    echo "Duration:    ${DURATION}s"
    echo "Results:     $METHOD/$DATASET/"
    echo "========================================"
    
    # === Log to timing summary file ===
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $DATASET - $method_name"
        echo "  Start:    $START_TIME"
        echo "  End:      $END_TIME"
        echo "  Duration: ${DURATION}s"
        echo "  Output:   $METHOD/$DATASET/"
        echo ""
    } >> "$TIMING_LOG"
    
    # === Also log to method-specific log ===
    local method_log="$METHOD/$DATASET/timing.log"
    {
        echo "Wavelet Basis Generation Timing"
        echo "=================================="
        echo "Dataset:    $DATASET"
        echo "Method:     $method_name"
        echo "Start time: $START_TIME"
        echo "End time:   $END_TIME"
        echo "Duration:   ${DURATION}s"
        echo "=================================="
    } > "$method_log"
    
    echo ""
    echo "Timing information saved to:"
    echo "  - $TIMING_LOG"
    echo "  - $method_log"
}

# === Route to appropriate method ===
case "${METHOD,,}" in
    baseline)
        # ============================================================================
        # Baseline MMF
        # ============================================================================
        echo "Running Baseline MMF on $DATASET..."
        echo "Start time: $START_TIME"
        echo ""
        
        PROGRAM="baseline_mmf_basis"
        setup_directories "$METHOD"
        
        NAME="$DATASET.$METHOD"
        
        python "$PROGRAM.py" \
            --data_folder="$DATA_FOLDER" \
            --dir="$METHOD" \
            --dataset="$DATASET" \
            --name="$NAME" \
            --dim="$DIM" \
            --seed=42
        
        log_completion "Baseline MMF"
        ;;
        
    random)
        # ============================================================================
        # Random MMF
        # ============================================================================
        echo "Running Random MMF on $DATASET..."
        echo "Start time: $START_TIME"
        echo ""
        
        PROGRAM="random_mmf_basis"
        setup_directories "$METHOD"
        
        NAME="$DATASET.$METHOD"
        
        python "$PROGRAM.py" \
            --data_folder="$DATA_FOLDER" \
            --dir="$METHOD" \
            --dataset="$DATASET" \
            --name="$NAME" \
            --K="$K" \
            --drop="$DROP" \
            --dim="$DIM" \
            --epochs="$EPOCHS" \
            --learning_rate="$LEARNING_RATE" \
            --seed=42
        
        log_completion "Random MMF"
        ;;
        
    k_neighbours)
        # ============================================================================
        # k-Neighbors MMF
        # ============================================================================
        echo "Running k-Neighbors MMF on $DATASET..."
        echo "Start time: $START_TIME"
        echo ""
        
        PROGRAM="k_neighbours_mmf_basis"
        setup_directories "$METHOD"
        
        NAME="$DATASET.$METHOD"
        
        python "$PROGRAM.py" \
            --data_folder="$DATA_FOLDER" \
            --dir="$METHOD" \
            --dataset="$DATASET" \
            --name="$NAME" \
            --K="$K" \
            --drop="$DROP" \
            --dim="$DIM" \
            --epochs="$EPOCHS" \
            --learning_rate="$LEARNING_RATE" \
            --seed=42
        
        log_completion "k-Neighbors MMF"
        ;;
        
    evolutionary_algorithm)
        # ============================================================================
        # Evolutionary Algorithm MMF
        # ============================================================================
        echo "Running Evolutionary Algorithm MMF on $DATASET..."
        echo "Start time: $START_TIME"
        echo ""
        
        PROGRAM="metaheuristics_mmf_basis"
        setup_directories "$METHOD"
        
        NAME="$DATASET.$METHOD"
        
        python "$PROGRAM.py" \
            --data_folder="$DATA_FOLDER" \
            --dir="$METHOD" \
            --dataset="$DATASET" \
            --name="$NAME" \
            --K="$K" \
            --dim="$DIM" \
            --method=ea \
            --epochs="$EPOCHS" \
            --learning_rate="$LEARNING_RATE" \
            --seed=42
        
        log_completion "Evolutionary Algorithm MMF"
        ;;
        
    directed_evolution)
        # ============================================================================
        # Directed Evolution MMF
        # ============================================================================
        echo "Running Directed Evolution MMF on $DATASET..."
        echo "Start time: $START_TIME"
        echo ""
        
        PROGRAM="metaheuristics_mmf_basis"
        setup_directories "$METHOD"
        
        NAME="$DATASET.$METHOD"
        
        python "$PROGRAM.py" \
            --data_folder="$DATA_FOLDER" \
            --dir="$METHOD" \
            --dataset="$DATASET" \
            --name="$NAME" \
            --K="$K" \
            --dim="$DIM" \
            --method=de \
            --epochs="$EPOCHS" \
            --learning_rate="$LEARNING_RATE" \
            --seed=42
        
        log_completion "Directed Evolution MMF"
        ;;
        
    *)
        echo "Error: Unknown method '$METHOD'"
        echo ""
        echo "Available methods:"
        echo "  baseline, random, k_neighbours, evolutionary_algorithm, directed_evolution"
        exit 1
        ;;
esac

# ============================================================================
# End
# ============================================================================
echo ""
echo "========================================"
echo "Execution Complete"
echo "========================================"
echo ""
echo "View timing summary: cat timing_logs/timing_summary.txt"

# === Deactivate conda environment ===
conda deactivate
