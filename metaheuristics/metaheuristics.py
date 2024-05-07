import torch
import random
import numpy as np
from learn_single_mmf import train_single_mmf


def get_nearest_indices(matrix, indices, K):
    N = matrix.size(1)
    L = len(indices)
    device = matrix.device
    
    # Compute the Euclidean distance between each row and the input indices row
    input_rows = matrix[indices]
    distances = torch.cdist(matrix, input_rows) 

    # Exclude the input indices from consideration
    distances[indices, torch.arange(L)] = float('inf')
    
    # Find the K nearest rows for each input index
    nearest_indices = torch.argsort(distances, dim=0)[:K]
    
    return nearest_indices.t()


def get_cost(matrix, indices, rest_indices, L, K):
    return train_single_mmf(matrix, L, K, indices.unsqueeze(-1).tolist(), rest_indices.tolist(), epochs=0, learning_rate=1e-1, early_stop=False)[3]


def crossover(parent1, parent2):
    parents = [parent1.clone().tolist(), parent2.clone().tolist()]
    L = len(parents[0])

    assert len(parents[0]) == len(set(parents[0])) and len(parents[1]) == len(set(parents[1]))

    duplicates = set()
    for v in parents[0]:
        if v in parents[1]:
            duplicates.add(v)

    crossover_ammount = random.randint(0, L - len(duplicates))
    idx1, idx2, ammount = 0, 0, 0

    while ammount < crossover_ammount:
        while idx1 < L and parents[0][idx1] in duplicates:
            idx1 += 1
        while idx2 < L and parents[1][idx2] in duplicates:
            idx2 += 1
        parents[0][idx1], parents[1][idx2] = parents[1][idx2], parents[0][idx1]
        idx1 += 1
        idx2 += 1
        ammount += 1

    child1, child2 = parents[0], parents[1]
        
    assert len(child1) == L and len(child2) == L
    assert len(child1) == len(set(child1)) and len(child2) == len(set(child2))
    return torch.tensor(child1, dtype=torch.int64), torch.tensor(child2, dtype=torch.int64) 


def evolutionary_algorithm(cost_function, matrix, L, K, population_size=1000, generations=100, mutation_rate=0.1):
    n = matrix.size(0)
    device = matrix.device
    
    # Initialization
    population = torch.stack([torch.randperm(n)[:L] for _ in range(population_size)], dim=0)
    nearest_indices = get_nearest_indices(matrix, torch.arange(n), K - 1)

    min_cost_per_gen = []
    mean_cost_per_gen = []
    all_time_min_cost_per_gen = []
    all_time_min_cost = float('inf')
    all_time_best_solution = None
    
    for gen in range(generations):
        # Evaluation
        costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
        min_cost_idx = torch.argmin(costs)
        best_solution = population[min_cost_idx]

        print(f'The best solution at generation {gen} is {torch.min(costs, dim=0)[0].item()}')
        min_cost_per_gen.append(torch.min(costs, dim=0)[0].item())
        mean_cost_per_gen.append(torch.mean(costs, dim=0).item())

        if min_cost_per_gen[-1] < all_time_min_cost:
            all_time_min_cost = min_cost_per_gen[-1]
            all_time_best_solution = best_solution

        all_time_min_cost_per_gen.append(all_time_min_cost)
        
        # Selection
        sorted_indices = torch.argsort(costs)
        selected_indices = sorted_indices[:population_size // 2]
        selected_population = population[selected_indices]

        for s in selected_population:
            assert len(s.tolist()) == len(set(s.tolist()))
        
        # Crossover
        offspring_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = torch.randperm(population_size // 2)[:2]
            child1, child2 = crossover(selected_population[parent1], selected_population[parent2])
            offspring_population.extend([child1, child2])
        
        # Mutation
        for i in range(len(offspring_population)):
            if torch.rand(1) < mutation_rate:
                mutation_indices = torch.randperm(L)[:2]  # Select 2 random mutation indices
                mutated_sample = offspring_population[i].clone()  # Make a copy of the sample
                # Swap the values at the mutation indices
                mutated_sample[mutation_indices[0]], mutated_sample[mutation_indices[1]] = \
                    mutated_sample[mutation_indices[1]].item(), mutated_sample[mutation_indices[0]].item()
                
                offspring_population[i] = mutated_sample
                assert len(offspring_population[i].tolist()) == len(set(offspring_population[i].tolist()))

            if torch.rand(1) < mutation_rate:
                mutation_idx = torch.randint(L, (1,))
                mutated_sample = offspring_population[i].clone()  # Make a copy of the sample
                pool = torch.from_numpy(np.setdiff1d(torch.arange(n).numpy(), mutated_sample.numpy()))
                random_index = torch.randint(len(pool), (1,))
                random_number = pool[random_index]
                mutated_sample[mutation_idx] = random_number.item()
                offspring_population[i] = mutated_sample
                assert len(offspring_population[i].tolist()) == len(set(offspring_population[i].tolist()))

        population = torch.stack(offspring_population)
    
    # Final evaluation
    final_costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
    min_cost_idx = torch.argmin(final_costs)
    best_solution = population[min_cost_idx]
    min_cost = final_costs[min_cost_idx]
    if min_cost < all_time_min_cost:
        all_time_min_cost = min_cost
        all_time_best_solution = best_solution
    all_time_min_cost_per_gen.append(all_time_min_cost)
    
    return all_time_best_solution, nearest_indices[all_time_best_solution, :], all_time_min_cost, min_cost_per_gen, mean_cost_per_gen, all_time_min_cost_per_gen