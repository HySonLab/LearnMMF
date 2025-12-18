import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams

# For datasets
from data_loader import *

# Models
from baseline_mmf_model import Baseline_MMF
from learnable_mmf_model import Learnable_MMF
from nystrom_model import nystrom_fps_model  # Using FPS version
from heuristics import *

# Set plot style
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 12

def _parse_args():
    parser = argparse.ArgumentParser(description='Compare MMF methods across dimensions')
    parser.add_argument('--dir', '-dir', type=str, default='./comparison_results', help='Output directory')
    parser.add_argument('--data_folder', '-data_folder', type=str, default='../data/', help='Data folder')
    parser.add_argument('--seed', '-s', type=int, default=123456789, help='Random seed')
    parser.add_argument('--device', '-device', type=str, default='cpu', help='cuda/cpu')
    parser.add_argument('--num_runs', '-num_runs', type=int, default=5, help='Number of runs for statistics')
    
    # Dimension ranges for each dataset
    parser.add_argument('--karate_dim_start', type=int, default=8, help='Karate starting dim')
    parser.add_argument('--karate_dim_end', type=int, default=24, help='Karate ending dim')
    parser.add_argument('--karate_dim_step', type=int, default=2, help='Karate dim step')
    
    parser.add_argument('--kron_dim_start', type=int, default=10, help='Kron starting dim')
    parser.add_argument('--kron_dim_end', type=int, default=80, help='Kron ending dim')
    parser.add_argument('--kron_dim_step', type=int, default=12, help='Kron dim step')
    
    parser.add_argument('--cayley_dim_start', type=int, default=10, help='Cayley starting dim')
    parser.add_argument('--cayley_dim_end', type=int, default=80, help='Cayley ending dim')
    parser.add_argument('--cayley_dim_step', type=int, default=12, help='Cayley dim step')
    parser.add_argument('--cayley_order', type=int, default=2, help='Cayley order')
    parser.add_argument('--cayley_depth', type=int, default=5, help='Cayley depth')
    
    # MMF parameters
    parser.add_argument('--L', '-L', type=int, default=2, help='Number of levels')
    parser.add_argument('--K', '-K', type=int, default=8, help='K for learnable MMF')
    parser.add_argument('--drop', '-drop', type=int, default=1, help='Drop rate for learnable MMF')
    parser.add_argument('--epochs', '-epochs', type=int, default=128, help='Epochs for learnable MMF')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='Learning rate for learnable MMF')
    
    args = parser.parse_args()
    return args

def test_baseline_mmf(A, L, dim, num_runs=5):
    """Test baseline MMF multiple times and return mean/std error"""
    N = A.size(0)
    errors = []
    norm = torch.norm(A, p='fro')
    
    for run in range(num_runs):
        model = Baseline_MMF(N, L, dim)
        A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model(A)
        
        loss = torch.norm(A - A_rec, p='fro')
        error = (loss / norm).item()
        errors.append(error)
    
    return np.mean(errors), np.std(errors)

def test_learnable_mmf(A, L, K, drop, dim, epochs=128, learning_rate=1e-4):
    """Test learnable MMF and return final error"""
    N = A.size(0)
    norm = torch.norm(A, p='fro')
    
    # Get wavelet indices using heuristics
    A_sparse = A.to_sparse()
    if drop == 1:
        wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(A_sparse, L, K, drop, dim)
    else:
        wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)
    
    # Create and train model
    model = Learnable_MMF(A, L, K, drop, dim, wavelet_indices, rest_indices)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    
    best_error = float('inf')
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()
        
        loss = torch.norm(A - A_rec, p='fro')
        loss.backward()
        
        error = (loss / norm).item()
        
        # Early stopping
        if error < best_error:
            best_error = error
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        
        # Cayley update for orthogonal matrices
        for l in range(L):
            X = torch.Tensor(model.all_O[l].data)
            G = torch.Tensor(model.all_O[l].grad.data)
            Z = torch.matmul(G, X.transpose(0, 1)) - torch.matmul(X, G.transpose(0, 1))
            tau = learning_rate
            Y = torch.matmul(torch.matmul(torch.inverse(torch.eye(K) + tau / 2 * Z), torch.eye(K) - tau / 2 * Z), X)
            model.all_O[l].data = Y.data
    
    return best_error

def test_nystrom_fps(A, dim, num_runs=5):
    """Test Nyström FPS multiple times and return mean/std error"""
    errors = []
    norm = torch.norm(A, p='fro')
    
    for run in range(num_runs):
        A_rec, C, W_inverse = nystrom_fps_model(A, dim)
        loss = torch.norm(A - A_rec, p='fro')
        error = (loss / norm).item()
        errors.append(error)
    
    return np.mean(errors), np.std(errors)

def plot_comparison(datasets_results, output_dir):
    """Plot comparison across all datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    dataset_names = ['Karate', 'Kron', 'Cayley']
    
    for idx, (dataset_name, results) in enumerate(zip(dataset_names, datasets_results)):
        ax = axes[idx]
        
        dims = results['dims']
        
        # Plot Baseline MMF
        baseline_mean = results['baseline_mean']
        baseline_std = results['baseline_std']
        ax.errorbar(dims, baseline_mean, yerr=baseline_std, marker='o', 
                   label='Baseline MMF', capsize=5, linewidth=2, markersize=8)
        
        # Plot Learnable MMF
        learnable_errors = results['learnable']
        ax.plot(dims, learnable_errors, marker='s', 
               label='Learnable MMF', linewidth=2, markersize=8)
        
        # Plot Nyström FPS
        nystrom_mean = results['nystrom_mean']
        nystrom_std = results['nystrom_std']
        ax.errorbar(dims, nystrom_mean, yerr=nystrom_std, marker='^', 
                   label='Nyström FPS', capsize=5, linewidth=2, markersize=8)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Normalized Error')
        ax.set_title(f'{dataset_name} Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_all_methods.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'comparison_all_methods.pdf'), bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir}")

def save_results_to_file(datasets_results, output_dir):
    """Save numerical results to text file"""
    dataset_names = ['Karate', 'Kron', 'Cayley']
    
    with open(os.path.join(output_dir, 'comparison_results.txt'), 'w') as f:
        for dataset_name, results in zip(dataset_names, datasets_results):
            f.write(f"\n{'='*60}\n")
            f.write(f"{dataset_name} Dataset Results\n")
            f.write(f"{'='*60}\n\n")
            
            dims = results['dims']
            f.write(f"{'Dim':<8} {'Baseline Mean':<15} {'Baseline Std':<15} "
                   f"{'Learnable':<15} {'Nyström Mean':<15} {'Nyström Std':<15}\n")
            f.write(f"{'-'*90}\n")
            
            for i, dim in enumerate(dims):
                f.write(f"{dim:<8} {results['baseline_mean'][i]:<15.6f} {results['baseline_std'][i]:<15.6f} "
                       f"{results['learnable'][i]:<15.6f} {results['nystrom_mean'][i]:<15.6f} "
                       f"{results['nystrom_std'][i]:<15.6f}\n")
    
    print(f"Saved results to {os.path.join(output_dir, 'comparison_results.txt')}")

def main():
    args = _parse_args()
    
    # Create output directory
    os.makedirs(args.dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device
    print(f"Using device: {device}")
    
    # Define datasets and their dimension ranges
    datasets_config = [
        {
            'name': 'karate',
            'dims': list(range(args.karate_dim_start, args.karate_dim_end + 1, args.karate_dim_step)),
            'loader': lambda: karate_def(data_folder=args.data_folder)
        },
        {
            'name': 'kron',
            'dims': list(range(args.kron_dim_start, args.kron_dim_end + 1, args.kron_dim_step)),
            'loader': lambda: kron_def()
        },
        {
            'name': 'cayley',
            'dims': list(range(args.cayley_dim_start, args.cayley_dim_end + 1, args.cayley_dim_step)),
            'loader': lambda: cayley_def(cayley_order=args.cayley_order, cayley_depth=args.cayley_depth)[0]
        }
    ]
    
    all_results = []
    
    for dataset_config in datasets_config:
        print(f"\n{'='*60}")
        print(f"Testing {dataset_config['name'].upper()} dataset")
        print(f"{'='*60}\n")
        
        # Load dataset
        A = dataset_config['loader']()
        N = A.size(0)
        print(f"Matrix size: {N} x {N}")
        
        dims = dataset_config['dims']
        results = {
            'dims': dims,
            'baseline_mean': [],
            'baseline_std': [],
            'learnable': [],
            'nystrom_mean': [],
            'nystrom_std': []
        }
        
        for dim in dims:
            print(f"\n--- Testing dimension: {dim} ---")
            
            # Test Baseline MMF
            print("  Testing Baseline MMF...")
            start_time = time.time()
            baseline_mean, baseline_std = test_baseline_mmf(A, args.L, dim, num_runs=args.num_runs)
            print(f"    Baseline MMF: {baseline_mean:.6f} ± {baseline_std:.6f} (time: {time.time()-start_time:.2f}s)")
            results['baseline_mean'].append(baseline_mean)
            results['baseline_std'].append(baseline_std)
            
            # Test Learnable MMF
            print("  Testing Learnable MMF...")
            start_time = time.time()
            learnable_error = test_learnable_mmf(A, args.L, args.K, args.drop, dim, 
                                                 epochs=args.epochs, learning_rate=args.learning_rate)
            print(f"    Learnable MMF: {learnable_error:.6f} (time: {time.time()-start_time:.2f}s)")
            results['learnable'].append(learnable_error)
            
            # Test Nyström FPS
            print("  Testing Nyström FPS...")
            start_time = time.time()
            nystrom_mean, nystrom_std = test_nystrom_fps(A, dim, num_runs=args.num_runs)
            print(f"    Nyström FPS: {nystrom_mean:.6f} ± {nystrom_std:.6f} (time: {time.time()-start_time:.2f}s)")
            results['nystrom_mean'].append(nystrom_mean)
            results['nystrom_std'].append(nystrom_std)
        
        all_results.append(results)
    
    # Plot and save results
    print(f"\n{'='*60}")
    print("Generating plots and saving results...")
    print(f"{'='*60}\n")
    
    plot_comparison(all_results, args.dir)
    save_results_to_file(all_results, args.dir)
    
    print("\nDone! All results saved.")

if __name__ == '__main__':
    main()