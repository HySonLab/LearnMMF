#!/bin/bash
# ==========================================
# Single Experiment Runner for Linux/Mac
# ==========================================

# Initialize conda for bash shell
echo "Initializing Anaconda environment..."

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
else
    echo "WARNING: Could not find conda installation"
fi

# Activate LearnMMF environment
echo "Activating LearnMMF environment..."
conda activate LearnMMF
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate LearnMMF environment"
    echo ""
    echo "Please ensure:"
    echo "  1. Anaconda/Miniconda is installed"
    echo "  2. LearnMMF environment exists"
    echo ""
    echo "To create the environment:"
    echo "  conda create -n LearnMMF python=3.8"
    echo "  conda activate LearnMMF"
    echo "  pip install torch numpy tqdm"
    echo ""
    exit 1
fi

echo ""
echo "=========================================="
echo "Running Single Experiment"
echo "=========================================="
echo ""

# Run the experiment with all provided arguments
python main.py "$@"
EXIT_CODE=$?

# Check if successful
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Experiment failed"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Experiment completed successfully!"
    echo "=========================================="
    echo ""
fi

# Deactivate environment
conda deactivate

exit $EXIT_CODE