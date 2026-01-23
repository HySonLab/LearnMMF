# Learnable Multiresolution Matrix Factorization (MMF)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/pytorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive implementation of **Multiresolution Matrix Factorization (MMF)** with learnable components, evolutionary metaheuristics optimization, and wavelet neural networks for graph analysis. This codebase supports matrix factorization research, graph classification, and node classification on citation networks.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Running Experiments](#running-experiments)
  - [Matrix Factorization](#1-matrix-factorization-experiments)
  - [Graph Classification](#2-graph-classification-experiments)
  - [Node Classification](#3-node-classification-experiments)
  - [Wavelet Visualization](#4-wavelet-visualization)
- [Datasets](#datasets)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

Multiresolution Matrix Factorization (MMF) is a unique approach to matrix factorization that does not rely on low-rank assumptions, making it particularly well-suited for modeling graphs with complex multiscale or hierarchical structures. This implementation provides:

- **Baseline MMF**: Original greedy algorithm with K=2 rotations
- **Learnable MMF**: Stiefel manifold optimization for arbitrary K
- **Metaheuristic MMF**: Evolutionary algorithms and directed evolution for index selection
- **Sparse MMF**: Memory-efficient implementation using sparse matrices
- **Wavelet Neural Networks**: Graph neural networks using MMF-derived wavelet bases

---

## Features

- Multiple MMF algorithms with different optimization strategies
- Support for various graph types (citation networks, molecular graphs, synthetic graphs)
- Comprehensive experiment runners with timing and logging
- Cross-platform support (Linux/macOS shell scripts and Windows batch files)
- Incremental saving for long-running experiments
- Visualization tools for wavelets on graphs
- Integration with conda environments for reproducibility

---

## Installation

### Prerequisites

- Python 3.7 or higher
- Conda (recommended) or pip
- Git

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/risilab/Learnable_MMF.git
cd Learnable_MMF

# Create and activate conda environment
conda env create -f environment.yml
conda activate LearnMMF
```

### Option 2: Manual Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install numpy tqdm matplotlib seaborn
```

### Verify Installation

```bash
cd source
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "from baseline_mmf_model import Baseline_MMF; print('MMF models loaded successfully')"
```

---

## Quick Start

### Run a Simple Matrix Factorization

```bash
cd source

# Factorize the Karate club network using baseline MMF
python baseline_mmf_run.py --dataset karate --L 26 --dim 8
```

### Generate Wavelet Basis for Molecular Graphs

```bash
cd experiments/graph_classification

# Generate baseline wavelet basis for MUTAG dataset
./generate_wavelet_basis.sh MUTAG baseline
```

### Train a Wavelet Neural Network

```bash
cd experiments/graph_classification

# Train classifier on MUTAG with baseline wavelets
./train_wavelet_network.sh MUTAG baseline
```

---

## Project Structure

```
Learnable_MMF/
├── data/                          # Datasets
│   ├── citeseer/                  # Citation graph
│   ├── cora/                      # Citation graph
│   ├── DD/                        # Protein dataset
│   ├── ENZYMES/                   # Protein structures
│   ├── karate/                    # Karate club network
│   ├── minnesota/                 # Road network
│   ├── mnist/                     # MNIST digit images
│   ├── MUTAG/                     # Mutagenic compounds
│   ├── NCI1/                      # Chemical compounds
│   ├── NCI109/                    # Chemical compounds
│   ├── PTC/                       # Chemical toxicity
│   └── WebKB/                     # Web pages
│
├── doc/                           # LaTeX documentation
│
├── experiments/
│   ├── graph_classification/      # Molecular graph classification
│   │   ├── generate_wavelet_basis.sh/.bat
│   │   ├── train_wavelet_network.sh/.bat
│   │   └── *.py                   # Python scripts
│   │
│   ├── matrix_factorization/      # Matrix factorization comparisons
│   │   ├── karate.py
│   │   ├── cayley.py
│   │   └── kron.py
│   │
│   └── node_classification/       # Citation graph node classification
│       ├── run_single.sh/.bat
│       └── main.py
│
├── source/                        # Core implementation
│   ├── baseline_mmf_model.py      # Original MMF (K=2)
│   ├── learnable_mmf_model.py     # Learnable MMF (arbitrary K)
│   ├── sparse_mmf_model.py        # Memory-efficient sparse MMF
│   ├── metaheuristics.py          # EA and DE optimization
│   ├── heuristics.py              # Index selection algorithms
│   ├── nystrom_model.py           # Nyström approximation baseline
│   ├── data_loader.py             # Dataset loading utilities
│   └── visualize_wavelets.py      # Wavelet visualization
│
├── environment.yml                # Conda environment specification
└── README.md                      # This file
```

---

## Core Components

### MMF Models

| Model | File | Description | Use Case |
|-------|------|-------------|----------|
| **Baseline MMF** | `baseline_mmf_model.py` | Original greedy algorithm with K=2 | Quick factorization, baseline comparison |
| **Learnable MMF** | `learnable_mmf_model.py` | Stiefel manifold optimization | Better quality with arbitrary K |
| **Sparse MMF** | `sparse_mmf_model.py` | COO sparse format operations | Large graphs, memory-constrained |
| **Smooth Wavelets** | `learnable_mmf_smooth_wavelets_model.py` | Wavelets with smoothness constraint | Smoother signal representation |

### Index Selection Algorithms

| Algorithm | Function | Description |
|-----------|----------|-------------|
| **Random** | `heuristics_random()` | Completely random selection |
| **K-Neighbors (Single)** | `heuristics_k_neighbors_single_wavelet()` | Graph-aware selection for drop=1 |
| **K-Neighbors (Multiple)** | `heuristics_k_neighbors_multiple_wavelets()` | Graph-aware selection for drop>1 |
| **Evolutionary Algorithm** | `evolutionary_algorithm()` | Population-based optimization |
| **Directed Evolution** | `directed_evolution()` | Focused mutation strategy |

---

## Running Experiments

### 1. Matrix Factorization Experiments

Compare different MMF methods on various matrices.

#### Karate Club Network

```bash
cd experiments/matrix_factorization

# Run full comparison (Baseline, Nyström, Random, K-neighbors, EA, DE)
python karate.py
```

**Output**: Generates `mmf_comparison_across_dimensions.pdf` showing reconstruction error vs. dimension.

#### Cayley Tree Graph

```bash
cd experiments/matrix_factorization

# Compare methods on Cayley tree (order=3, depth=4, N=161 nodes)
python cayley.py
```

#### Kronecker Product Matrix

```bash
cd experiments/matrix_factorization

# Large synthetic matrix (512 x 512)
python kron.py
```

#### Batch Experiments with Baseline MMF

```bash
cd source

# Run baseline MMF on various datasets
./baseline_mmf_run.sh

# Or run manually with custom parameters
python baseline_mmf_run.py \
    --dataset karate \
    --L 26 \
    --dim 8 \
    --num_times 10 \
    --device cpu
```

#### Learnable MMF Experiments

```bash
cd source

# Run learnable MMF with Stiefel optimization
./learnable_mmf_run.sh

# Or run manually
python learnable_mmf_run.py \
    --dataset cayley \
    --L 145 \
    --K 8 \
    --drop 1 \
    --dim 16 \
    --epochs 100000 \
    --learning_rate 1e-3
```

---

### 2. Graph Classification Experiments

Train wavelet neural networks on molecular graph datasets.

#### Step 1: Generate Wavelet Basis

Choose a dataset and method:

**Linux/macOS:**
```bash
cd experiments/graph_classification

# Syntax: ./generate_wavelet_basis.sh [DATASET] [METHOD]

# Available datasets: MUTAG, PTC, DD, NCI1
# Available methods: baseline, random, k_neighbours, evolutionary_algorithm, directed_evolution

# Examples:
./generate_wavelet_basis.sh MUTAG baseline
./generate_wavelet_basis.sh PTC evolutionary_algorithm
./generate_wavelet_basis.sh DD directed_evolution
./generate_wavelet_basis.sh NCI1 k_neighbours
```

**Windows:**
```cmd
cd experiments\graph_classification

REM Syntax: generate_wavelet_basis.bat [DATASET] [METHOD]
generate_wavelet_basis.bat MUTAG baseline
generate_wavelet_basis.bat PTC evolutionary_algorithm
```

**Output Structure:**
```
[METHOD]/
└── [DATASET]/
    ├── [DATASET].[METHOD].adjs.pt
    ├── [DATASET].[METHOD].laplacians.pt
    ├── [DATASET].[METHOD].mother_coeffs.pt
    ├── [DATASET].[METHOD].father_coeffs.pt
    ├── [DATASET].[METHOD].mother_wavelets.pt
    ├── [DATASET].[METHOD].father_wavelets.pt
    └── timing.log
```

#### Step 2: Train Wavelet Neural Network

**Linux/macOS:**
```bash
cd experiments/graph_classification

# Syntax: ./train_wavelet_network.sh [DATASET] [METHOD]
./train_wavelet_network.sh MUTAG baseline
./train_wavelet_network.sh PTC evolutionary_algorithm
```

**Windows:**
```cmd
cd experiments\graph_classification
train_wavelet_network.bat MUTAG baseline
```

**Training performs 10-fold cross-validation** and outputs:
- Individual split results
- Mean accuracy ± standard deviation
- Best model checkpoints

#### Advanced: Metaheuristic Wavelet Generation

For large datasets with incremental saving and resume capability:

```bash
cd experiments/graph_classification

# Process specific range of molecules (useful for parallel execution)
python metaheuristics_mmf_basis.py \
    --dataset DD \
    --method ea \
    --start_idx 0 \
    --end_idx 100 \
    --sort_by_size \
    --resume

# Consolidate incremental results into single files
python consolidate_incremental_results.py \
    --input_dir evolutionary_algorithm/DD \
    --name DD.evolutionary_algorithm
```

#### Dataset Statistics Analysis

```bash
cd experiments/graph_classification

# Generate comprehensive statistics and visualizations
python dataset_statistics.py \
    --dataset DD \
    --output_dir ./statistics \
    --format pdf
```

**Generates:**
- Size distribution histograms
- Degree statistics
- Class distribution analysis
- Large graph analysis (>200 atoms)

---

### 3. Node Classification Experiments

Semi-supervised node classification on citation graphs (Cora, Citeseer).

#### Basic Usage

**Linux/macOS:**
```bash
cd experiments/node_classification

# Run with default settings on Cora
./run_single.sh --dataset cora --split MMF1

# Run on Citeseer with custom parameters
./run_single.sh \
    --dataset citeseer \
    --split MMF2 \
    --K 16 \
    --dim 200 \
    --num-layers 6 \
    --hidden-dim 100 \
    --wnn-epochs 256
```

**Windows:**
```cmd
cd experiments\node_classification
run_single.bat --dataset cora --split MMF1
```

#### Data Split Options

| Split | Train | Validation | Test |
|-------|-------|------------|------|
| MMF1 | 20% | 20% | 60% |
| MMF2 | 40% | 20% | 40% |
| MMF3 | 60% | 20% | 20% |

#### Advanced Configuration

```bash
python main.py \
    --dataset cora \
    --data-folder ../../data/ \
    --K 16 \
    --L 2000 \
    --dim 708 \
    --heuristics smart \
    --mmf-epochs 10000 \
    --mmf-lr 1e-4 \
    --num-layers 6 \
    --hidden-dim 100 \
    --wnn-epochs 256 \
    --wnn-lr 1e-3 \
    --split MMF1 \
    --seed 42 \
    --output-dir ./output \
    --save-mmf \
    --save-wavelets
```

#### Using Pre-computed Wavelets

```bash
# Skip MMF training and use saved wavelets
python main.py \
    --dataset cora \
    --load-wavelets ./output/wavelets \
    --skip-mmf \
    --wnn-epochs 256
```

---

### 4. Wavelet Visualization

Visualize wavelets on graphs with 3D plots.

#### Basic Visualization

```bash
cd source

# Visualize wavelets on MUTAG molecules
python visualize_wavelets.py \
    --dataset MUTAG \
    --data-folder ../data/ \
    --molecule-id 0 \
    --layout kamada_kawai \
    --save-wavelets mother \
    --output-folder ./visualizations
```

#### Multiple Molecules

```bash
# Process a range of molecules
python visualize_wavelets.py \
    --dataset MUTAG \
    --molecule-range "0-10" \
    --layout fruchterman_reingold \
    --save-wavelets both \
    --wavelet-range "0-5"
```

#### Layout Options

| Layout | Description | Best For |
|--------|-------------|----------|
| `kamada_kawai` | Force-directed, aesthetic | Most graphs |
| `fruchterman_reingold` | Force-directed, fast | Large graphs |
| `spectral` | Laplacian eigenvectors | Structured graphs |
| `circular` | Nodes in circle | Cycles, rings |
| `shell` | Concentric circles | Hierarchical graphs |
| `spiral` | Spiral arrangement | Sequential data |

#### Programmatic Usage

```python
from visualize_wavelets import draw_graph_with_wavelet, get_graph_layout

# Load or create adjacency matrix and wavelet
A = torch.zeros(10, 10)  # Your adjacency matrix
wavelet = torch.randn(1, 10)  # Your wavelet

# Get layout
nodes_x, nodes_y = get_graph_layout(A.numpy(), layout='kamada_kawai')

# Draw and save
draw_graph_with_wavelet(
    A, wavelet, 'my_wavelet',
    nodes_x=nodes_x, nodes_y=nodes_y,
    title='My Wavelet Visualization'
)
```

---

## Datasets

### Included Datasets

| Dataset | Type | Nodes/Graphs | Description |
|---------|------|--------------|-------------|
| **Cora** | Citation | 2,708 nodes | Paper citation network |
| **Citeseer** | Citation | 3,312 nodes | Paper citation network |
| **WebKB** | Web | 877 pages | Scientific publications |
| **MUTAG** | Molecular | 188 graphs | Mutagenic compounds |
| **PTC** | Molecular | 344 graphs | Chemical toxicity |
| **DD** | Molecular | 1,113 graphs | Protein structures |
| **NCI1** | Molecular | 4,110 graphs | Cancer screening |
| **NCI109** | Molecular | 4,127 graphs | Cancer screening |
| **ENZYMES** | Molecular | 600 graphs | Protein structures |
| **Karate** | Social | 34 nodes | Karate club network |
| **Minnesota** | Road | 2,642 nodes | Road network |
| **MNIST** | Image | 1,000 samples | RBF kernel matrix |

### Synthetic Datasets

| Dataset | Function | Parameters |
|---------|----------|------------|
| **Kronecker** | `kron_def()` | 512×512 matrix |
| **Cycle** | `cycle_def()` | 64-node cycle |
| **Cayley** | `cayley_def(order, depth)` | Configurable tree |

### Loading Custom Data

```python
from data_loader import citation_def, karate_def, cayley_def

# Citation graph
adj, L_norm, features, labels, ids, label_names = citation_def(
    data_folder='../data/',
    dataset='cora'
)

# Karate club
L_norm = karate_def(data_folder='../data/')

# Cayley tree
L_norm, x_coords, y_coords, edges = cayley_def(
    cayley_order=3,
    cayley_depth=4
)
```

---

## Configuration Reference

### MMF Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `N` | Matrix dimension | Auto from data |
| `L` | Number of wavelet levels | N - dim |
| `K` | Rotation matrix size | 2-16 |
| `drop` | Columns dropped per level | 1 |
| `dim` | Father wavelet count | 2-64 |

### Optimization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Training iterations | 1000-100000 |
| `learning_rate` | Stiefel manifold LR | 1e-3 to 1e-4 |
| `early_stop` | Stop on convergence | True |
| `opt` | Optimizer type | 'original' |

### Metaheuristic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `population_size` | EA/DE population | 50-100 |
| `generations` | Evolution iterations | 100 |
| `mutation_rate` | EA mutation probability | 0.2 |
| `sample_kept_rate` | DE selection ratio | 0.3 |

### Neural Network Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_layers` | Spectral convolution layers | 6 |
| `hidden_dim` | Hidden dimension | 32-100 |
| `batch_size` | Training batch size | 20 |

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Use CPU for large graphs
python script.py --device cpu

# Or reduce batch size
python script.py --batch_size 10
```

#### Environment Activation Fails

```bash
# Initialize conda for your shell
conda init bash  # or zsh, fish, etc.
source ~/.bashrc

# Recreate environment if needed
conda env remove -n LearnMMF
conda env create -f environment.yml
```

#### Missing Wavelet Basis Files

```bash
# Generate basis before training
./generate_wavelet_basis.sh MUTAG baseline

# Check files exist
ls baseline/MUTAG/
```

#### Graph Too Large

```bash
# Use sparse implementation for large graphs
python sparse_mmf_run.py --dataset cora --K 16 --heuristics smart

# Or use incremental processing
python metaheuristics_mmf_basis.py \
    --dataset DD \
    --start_idx 0 \
    --end_idx 100 \
    --resume
```

### Performance Tips

1. **Use smaller K for initial experiments** (K=8 is often sufficient)
2. **Enable early stopping** to save computation time
3. **Sort molecules by size** for efficient batch processing
4. **Use incremental saving** for long-running experiments
5. **Monitor timing logs** to identify bottlenecks

---

## Citation

If you use this code in your research, please cite our papers:

```bibtex
@InProceedings{pmlr-v196-hy22a,
  title     = {Multiresolution Matrix Factorization and Wavelet Networks on Graphs},
  author    = {Hy, Truong Son and Kondor, Risi},
  booktitle = {Proceedings of Topological, Algebraic, and Geometric Learning Workshops 2022},
  pages     = {172--182},
  year      = {2022},
  volume    = {196},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v196/hy22a.html}
}

@misc{hy2024learning,
  title         = {Learning to Solve Multiresolution Matrix Factorization by Manifold 
                   Optimization and Evolutionary Metaheuristics},
  author        = {Truong Son Hy and Thieu Khang and Risi Kondor},
  year          = {2024},
  eprint        = {2406.00469},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@misc{hy2021learning,
  title         = {Learning Multiresolution Matrix Factorization and its Wavelet 
                   Networks on Graphs},
  author        = {Truong Son Hy and Risi Kondor},
  year          = {2021},
  eprint        = {2111.01940},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.