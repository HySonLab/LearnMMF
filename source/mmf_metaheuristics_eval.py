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
    """Run a single experiment and return all costs and timings"""
    # Set unique seed for this experiment
    experiment_seed = base_seed + experiment_id
    set_seed(experiment_seed)
    
    results = {}
    timings = {}
    
    # Baseline (original) MMF
    start = time.time()
    original_mmf = Baseline_MMF(N=N, L=N - 8, dim=8)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = original_mmf(karate_laplacian)
    results['original_cost'] = torch.norm(karate_laplacian - A_rec, p='fro').item()
    timings['original_time'] = time.time() - start
    
    # Learnable MMF (random indices)
    start = time.time()
    wavelet_indices, rest_indices = heuristics_random(karate_laplacian.to_sparse(), L=26, K=8, drop=1, dim=8)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(
        karate_laplacian, L=26, K=8, drop=1, dim=8, 
        wavelet_indices=wavelet_indices, rest_indices=rest_indices, 
        epochs=0, learning_rate=1e-4, early_stop=True
    )
    results['random_cost'] = torch.norm(karate_laplacian - A_rec, p='fro').item()
    timings['random_time'] = time.time() - start
    
    # Learnable MMF (heuristics to select indices)
    start = time.time()
    wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(karate_laplacian.to_sparse(), L=26, K=8, drop=1, dim=8)
    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(
        karate_laplacian, L=26, K=8, drop=1, dim=8, 
        wavelet_indices=wavelet_indices, rest_indices=rest_indices, 
        epochs=0, learning_rate=1e-3, early_stop=True
    )
    results['heuristics_cost'] = torch.norm(karate_laplacian - A_rec, p='fro').item()
    timings['heuristics_time'] = time.time() - start
    
    # Learnable MMF (EA to select indices)
    start = time.time()
    wavelet_indices, rest_indices, ea_cost, ea_min_cost_per_gen, ea_mean_cost_per_gen, ea_all_time_min_cost_per_gen = evolutionary_algorithm(
        get_cost, karate_laplacian, L=26, K=8, 
        population_size=20, generations=100, mutation_rate=0.2
    )
    results['ea_all_time_min'] = ea_all_time_min_cost_per_gen
    timings['ea_time'] = time.time() - start
    
    # Learnable MMF (DE to select indices)
    start = time.time()
    wavelet_indices, rest_indices, de_cost, de_min_cost_per_gen, de_mean_cost_per_gen, de_all_time_min_cost_per_gen = directed_evolution(
        get_cost, karate_laplacian, L=26, K=8, 
        population_size=10, generations=100, sample_kept_rate=0.3
    )
    results['de_all_time_min'] = de_all_time_min_cost_per_gen
    timings['de_time'] = time.time() - start
    
    # Combine results and timings
    results.update(timings)
    return results

def main():
    # Set base random seed for reproducibility
    BASE_SEED = 42  # Change this value to get different experiment variations
    
    # Set initial seed
    set_seed(BASE_SEED)
    
    start_time = time.time()
    
    # Configuration
    num_experiments = 10  # Adjust this number based on your needs
    
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
    generation = 100
    
    # Extract costs for each method
    original_costs = np.array([r['original_cost'] for r in all_results])
    random_costs = np.array([r['random_cost'] for r in all_results])
    heuristics_costs = np.array([r['heuristics_cost'] for r in all_results])
    
    # Extract timings for each method
    original_times = np.array([r['original_time'] for r in all_results])
    random_times = np.array([r['random_time'] for r in all_results])
    heuristics_times = np.array([r['heuristics_time'] for r in all_results])
    ea_times = np.array([r['ea_time'] for r in all_results])
    de_times = np.array([r['de_time'] for r in all_results])
    
    # Extract EA and DE trajectories
    ea_trajectories = np.array([r['ea_all_time_min'] for r in all_results])
    de_trajectories = np.array([r['de_all_time_min'] for r in all_results])
    
    # Calculate statistics for baseline methods
    original_mean = np.mean(original_costs)
    original_std = np.std(original_costs)
    
    random_mean = np.mean(random_costs)
    random_std = np.std(random_costs)
    
    heuristics_mean = np.mean(heuristics_costs)
    heuristics_std = np.std(heuristics_costs)
    
    # Calculate statistics for EA and DE
    ea_mean = np.mean(ea_trajectories, axis=0)
    ea_std = np.std(ea_trajectories, axis=0)
    
    de_mean = np.mean(de_trajectories, axis=0)
    de_std = np.std(de_trajectories, axis=0)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    generations = range(generation)
    
    # Plot Original MMF with confidence interval
    plt.plot(generations, [original_mean] * generation, 
             label='Original MMF', linewidth=2)
    plt.fill_between(generations, 
                     [original_mean - original_std] * generation, 
                     [original_mean + original_std] * generation, 
                     alpha=0.2)
    
    # Plot Random indices with confidence interval
    plt.plot(generations, [random_mean] * generation, 
             label='Random indices', linewidth=2)
    plt.fill_between(generations, 
                     [random_mean - random_std] * generation, 
                     [random_mean + random_std] * generation, 
                     alpha=0.2)
    
    # Plot K neighbours heuristics with confidence interval
    plt.plot(generations, [heuristics_mean] * generation, 
             label='K neighbours heuristics', linewidth=2)
    plt.fill_between(generations, 
                     [heuristics_mean - heuristics_std] * generation, 
                     [heuristics_mean + heuristics_std] * generation, 
                     alpha=0.2)
    
    # Plot EA with confidence interval
    plt.plot(generations, ea_mean, label='Best EA evaluation', linewidth=2)
    plt.fill_between(generations, ea_mean - ea_std, ea_mean + ea_std, 
                     alpha=0.2)
    
    # Plot DE with confidence interval
    plt.plot(generations, de_mean, label='Best DE evaluation', linewidth=2)
    plt.fill_between(generations, de_mean - de_std, de_mean + de_std, 
                     alpha=0.2)
    
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
    print(f"\nCosts (mean ± std):")
    print(f"Original MMF: {original_mean:.4f} ± {original_std:.4f}")
    print(f"Random indices: {random_mean:.4f} ± {random_std:.4f}")
    print(f"K neighbours heuristics: {heuristics_mean:.4f} ± {heuristics_std:.4f}")
    print(f"EA final: {ea_mean[-1]:.4f} ± {ea_std[-1]:.4f}")
    print(f"DE final: {de_mean[-1]:.4f} ± {de_std[-1]:.4f}")
    
    print(f"\nAverage Execution Times:")
    print(f"Original MMF: {np.mean(original_times):.4f} ± {np.std(original_times):.4f} seconds")
    print(f"Random indices: {np.mean(random_times):.4f} ± {np.std(random_times):.4f} seconds")
    print(f"K neighbours heuristics: {np.mean(heuristics_times):.4f} ± {np.std(heuristics_times):.4f} seconds")
    print(f"EA: {np.mean(ea_times):.4f} ± {np.std(ea_times):.4f} seconds")
    print(f"DE: {np.mean(de_times):.4f} ± {np.std(de_times):.4f} seconds")
    
    # Save results to file for reproducibility verification
    results_dict = {
        'base_seed': BASE_SEED,
        'num_experiments': num_experiments,
        'original_costs': original_costs.tolist(),
        'random_costs': random_costs.tolist(),
        'heuristics_costs': heuristics_costs.tolist(),
        'ea_trajectories': ea_trajectories.tolist(),
        'de_trajectories': de_trajectories.tolist(),
        'original_times': original_times.tolist(),
        'random_times': random_times.tolist(),
        'heuristics_times': heuristics_times.tolist(),
        'ea_times': ea_times.tolist(),
        'de_times': de_times.tolist()
    }
    
    import json
    with open(f'experiment_results_seed_{BASE_SEED}.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to experiment_results_seed_{BASE_SEED}.json")
    
    # Save plot as PDF
    pdf_filename = f'experiment_plot_seed_{BASE_SEED}.pdf'
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {pdf_filename}")

    # Optionally display the plot (comment out if you don't want to show it)
    plt.show()

if __name__ == '__main__':
    main()