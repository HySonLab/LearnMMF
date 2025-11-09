import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from tqdm import tqdm
from heuristics import *
from data_loader import *
from learnable_mmf_model import *
from baseline_mmf_model import Baseline_MMF
from metaheuristics import evolutionary_algorithm, get_cost, directed_evolution

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_single_experiment(experiment_id, karate_laplacian, N, base_seed):
    """Run a single experiment and return all costs"""
    # Set unique seed for this experiment
    experiment_seed = base_seed + experiment_id
    set_seed(experiment_seed)
    
    results = {}
    
    # Baseline (original) MMF
    original_mmf = Baseline_MMF(N=N, L=N - 8, dim=8)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = original_mmf(karate_laplacian)
    results['original_cost'] = torch.norm(karate_laplacian - A_rec, p='fro').item()
    
    # Learnable MMF (random indices)
    wavelet_indices, rest_indices = heuristics_random(karate_laplacian.to_sparse(), L=26, K=8, drop=1, dim=8)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(
        karate_laplacian, L=26, K=8, drop=1, dim=8, 
        wavelet_indices=wavelet_indices, rest_indices=rest_indices, 
        epochs=0, learning_rate=1e-4, early_stop=True
    )
    results['random_cost'] = torch.norm(karate_laplacian - A_rec, p='fro').item()
    
    # Learnable MMF (heuristics to select indices)
    wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(karate_laplacian.to_sparse(), L=26, K=8, drop=1, dim=8)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(
        karate_laplacian, L=26, K=8, drop=1, dim=8, 
        wavelet_indices=wavelet_indices, rest_indices=rest_indices, 
        epochs=0, learning_rate=1e-3, early_stop=True
    )
    results['heuristics_cost'] = torch.norm(karate_laplacian - A_rec, p='fro').item()
    
    # Learnable MMF (EA to select indices)
    wavelet_indices, rest_indices, ea_cost, ea_min_cost_per_gen, ea_mean_cost_per_gen, ea_all_time_min_cost_per_gen = evolutionary_algorithm(
        get_cost, karate_laplacian, L=26, K=8, 
        population_size=100, generations=10, mutation_rate=0.2
    )
    results['ea_all_time_min'] = ea_all_time_min_cost_per_gen
    
    # Learnable MMF (DE to select indices)
    wavelet_indices, rest_indices, de_cost, de_min_cost_per_gen, de_mean_cost_per_gen, de_all_time_min_cost_per_gen = directed_evolution(
        get_cost, karate_laplacian, L=26, K=8, 
        population_size=10, generations=10, sample_kept_rate=0.3
    )
    results['de_all_time_min'] = de_all_time_min_cost_per_gen
    
    return results

def main():
    # Set base random seed for reproducibility
    BASE_SEED = 42  # Change this value to get different experiment variations
    
    # Set initial seed
    set_seed(BASE_SEED)
    
    start_time = time.time()
    
    # Configuration
    num_experiments = 3  # Adjust this number based on your needs
    
    # Data loading
    karate_laplacian = karate_def(r'C:\Users\Khang Nguyen\Documents\GitHub\LearnMMF\data')
    N = karate_laplacian.size(0)
    
    print(f"Running {num_experiments} experiments consecutively with base seed {BASE_SEED}...")
    print(f"Experiment seeds: {[BASE_SEED + i for i in range(num_experiments)]}")
    
    # Run experiments consecutively with progress bar
    all_results = []
    for i in tqdm(range(num_experiments), desc="Experiments", unit="exp"):
        try:
            result = run_single_experiment(i, karate_laplacian, N, BASE_SEED)
            all_results.append(result)
        except Exception as e:
            print(f"\nExperiment {i+1} failed with error: {e}")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\nTotal runtime: {total_runtime:.2f} seconds")
    print(f"Average time per experiment: {total_runtime/num_experiments:.2f} seconds")
    
    # Process results
    generation = 10
    
    # Extract costs for each method
    original_costs = [r['original_cost'] for r in all_results]
    random_costs = [r['random_cost'] for r in all_results]
    heuristics_costs = [r['heuristics_cost'] for r in all_results]
    
    # Extract EA and DE trajectories
    ea_trajectories = np.array([r['ea_all_time_min'] for r in all_results])
    de_trajectories = np.array([r['de_all_time_min'] for r in all_results])
    
    # Calculate statistics
    original_mean = np.mean(original_costs)
    random_mean = np.mean(random_costs)
    heuristics_mean = np.mean(heuristics_costs)
    
    ea_mean = np.mean(ea_trajectories, axis=0)
    ea_std = np.std(ea_trajectories, axis=0)
    
    de_mean = np.mean(de_trajectories, axis=0)
    de_std = np.std(de_trajectories, axis=0)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    generations = range(generation)
    
    # Plot baseline methods (constant lines)
    plt.plot(generations, [original_mean] * generation, 
             label='Original MMF', linestyle='--', linewidth=2)
    plt.plot(generations, [random_mean] * generation, 
             label='Random indices', linestyle='--', linewidth=2)
    plt.plot(generations, [heuristics_mean] * generation, 
             label='K neighbours heuristics', linestyle='--', linewidth=2)
    
    # Plot EA with confidence interval
    plt.plot(generations, ea_mean, label='Best EA evaluation', linewidth=2)
    plt.fill_between(generations, ea_mean - ea_std, ea_mean + ea_std, 
                     alpha=0.3, label='EA ±1 std')
    
    # Plot DE with confidence interval
    plt.plot(generations, de_mean, label='Best DE evaluation', linewidth=2)
    plt.fill_between(generations, de_mean - de_std, de_mean + de_std, 
                     alpha=0.3, label='DE ±1 std')
    
    # Labels and formatting
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title(f'Metaheuristics versus baselines (n={num_experiments} experiments, seed={BASE_SEED})', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Base Seed: {BASE_SEED}")
    print(f"Original MMF: {original_mean:.4f} ± {np.std(original_costs):.4f}")
    print(f"Random indices: {random_mean:.4f} ± {np.std(random_costs):.4f}")
    print(f"K neighbours heuristics: {heuristics_mean:.4f} ± {np.std(heuristics_costs):.4f}")
    print(f"EA final: {ea_mean[-1]:.4f} ± {ea_std[-1]:.4f}")
    print(f"DE final: {de_mean[-1]:.4f} ± {de_std[-1]:.4f}")
    
    # Save results to file for reproducibility verification
    results_dict = {
        'base_seed': BASE_SEED,
        'num_experiments': num_experiments,
        'original_costs': original_costs,
        'random_costs': random_costs,
        'heuristics_costs': heuristics_costs,
        'ea_trajectories': ea_trajectories.tolist(),
        'de_trajectories': de_trajectories.tolist()
    }
    
    import json
    with open(f'experiment_results_seed_{BASE_SEED}.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to experiment_results_seed_{BASE_SEED}.json")
    
    # Display plot
    plt.show()
    
    # Keep the plot window open until the user exits
    while True:
        plt.pause(0.05)

if __name__ == '__main__':
    main()