import sys
import torch
import json
sys.path.append('../../source/')
from data_loader import *
from baseline_mmf_model import Baseline_MMF
from nystrom_model import nystrom_fps_model
from heuristics import *
from learnable_mmf_model import *
from metaheuristics import evolutionary_algorithm, directed_evolution, get_cost_numpy_float32, get_cost
import matplotlib.pyplot as plt
import random
import numpy as np

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

karate_laplacian = karate_def('D:\codebase\Learnable_MMF\data')
N = karate_laplacian.size(0)

results_karate = {
    'Original MMF': [],
    'Nystrom': [],
    'Random MMF': [],
    'K neighbours MMF': [],
    'EA MMF': [],
    'DE MMF': [],
}

for column in range(8, 25, 4):
    print(f"\n=== Processing dimension: {column} ===")
    
    # Original MMF
    print("Running Original MMF...")
    model = Baseline_MMF(N, N - column, column) # N = L * drop + dim
    A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model(karate_laplacian.clone())
    diff_rec = torch.abs(karate_laplacian - A_rec)
    loss = torch.norm(diff_rec, p = 'fro')
    results_karate['Original MMF'].append([column, loss.item()])

    # Nystrom
    print("Running Nystrom...")
    A_rec, C, W_inverse = nystrom_fps_model(karate_laplacian.clone(), dim = column)
    results_karate['Nystrom'].append([column, torch.norm(karate_laplacian - A_rec, p = 'fro').item()])

    # Random MMF
    print("Running Random MMF...")
    wavelet_indices, rest_indices = heuristics_random(karate_laplacian.to_sparse(), L = column, K = 8, drop = 1, dim = N - column)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian.clone(), L = column, K = 8, drop = 1, dim = N - column, wavelet_indices = wavelet_indices, rest_indices = rest_indices, epochs = 64, learning_rate = 1e-4, early_stop = True)  
    results_karate['Random MMF'].append([column, torch.norm(karate_laplacian - A_rec, p = 'fro').item()])

    # K neighbours 
    print("Running K neighbours MMF...")
    wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(karate_laplacian.to_sparse(), L = column, K = 8, drop = 1, dim = N - column)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian.clone(), L = column, K = 8, drop = 1, dim = N - column, wavelet_indices = wavelet_indices, rest_indices = rest_indices, epochs = 64, learning_rate = 1e-4, early_stop = True)
    results_karate['K neighbours MMF'].append([column, torch.norm(karate_laplacian - A_rec, p = 'fro').item()])
    
    # EA MMF
    print("Running EA MMF...")
    wavelet_indices, rest_indices, ea_cost, ea_min_cost_per_gen, ea_mean_cost_per_gen, ea_all_time_min_cost_per_gen = evolutionary_algorithm(
        get_cost, karate_laplacian.clone(), L = column, K = 8, 
        population_size = 50, generations = 100, mutation_rate = 0.2
    )
    print(f'EA loss {ea_all_time_min_cost_per_gen[-1]}')
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian.clone(), L = column, K = 8, drop = 1, dim = N - column, wavelet_indices = wavelet_indices.unsqueeze(-1).tolist(), rest_indices = rest_indices.tolist(), epochs = 64, learning_rate = 1e-4, early_stop = True)
    results_karate['EA MMF'].append([column, torch.norm(karate_laplacian - A_rec, p = 'fro').item()])

    
    # DE MMF
    print("Running DE MMF...")
    wavelet_indices, rest_indices, de_cost, de_min_cost_per_gen, de_mean_cost_per_gen, de_all_time_min_cost_per_gen = directed_evolution(
        get_cost, karate_laplacian.clone(), L = column, K = 8, 
        population_size = 50, generations = 100, sample_kept_rate = 0.3
    )
    print(f'DE loss {de_all_time_min_cost_per_gen[-1]}')
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian.clone(), L = column, K = 8, drop = 1, dim = N - column, wavelet_indices = wavelet_indices.unsqueeze(-1).tolist(), rest_indices = rest_indices.tolist(), epochs = 64, learning_rate = 1e-4, early_stop = True)
    results_karate['DE MMF'].append([column, torch.norm(karate_laplacian - A_rec, p = 'fro').item()])

# Extract dimensions and errors for plotting
dimensions = [r[0] for r in results_karate['Original MMF']]
original_errors = [r[1] for r in results_karate['Original MMF']]
nystrom_errors = [r[1] for r in results_karate['Nystrom']]
random_errors = [r[1] for r in results_karate['Random MMF']]
k_neighbours_errors = [r[1] for r in results_karate['K neighbours MMF']]
ea_errors = [r[1] for r in results_karate['EA MMF']]
de_errors = [r[1] for r in results_karate['DE MMF']]

# Create plot
plt.figure(figsize=(12, 7))

plt.plot(dimensions, original_errors, marker='o', linewidth=2, markersize=8, label='Original MMF (K = 2)')
plt.plot(dimensions, nystrom_errors, marker='s', linewidth=2, markersize=8, label='Nystrom')
plt.plot(dimensions, random_errors, marker='^', linewidth=2, markersize=8, label='Random MMF (K = 8)')
plt.plot(dimensions, k_neighbours_errors, marker='d', linewidth=2, markersize=8, label='K neighbours MMF (K = 8)')
plt.plot(dimensions, ea_errors, marker='*', linewidth=2, markersize=10, label='EA MMF (K = 8)')
plt.plot(dimensions, de_errors, marker='P', linewidth=2, markersize=8, label='DE MMF (K = 8)')

plt.xlabel('Number of columns', fontsize=12)
plt.ylabel('Frobenius Norm Error', fontsize=12)
plt.title('Karate Club Laplacian Matrix (N = 34)', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('mmf_comparison_across_dimensions.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('mmf_comparison_across_dimensions.png', format='png', dpi=300, bbox_inches='tight')
print("\nPlots saved to mmf_comparison_across_dimensions.pdf and .png")

# Print summary
print("\n=== Summary of Results ===")
for method in results_karate:
    print(f"\n{method}:")
    for dim, error in results_karate[method]:
        print(f"  Dimension {dim}: Error = {error:.6f}")

plt.show()

# Save raw output to a JSON file
with open('karate_results.json', 'w') as f:
    json.dump(results_karate, f, indent=4)
    
print("Raw data saved to karate_results.json")