import torch


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
    return 0


def evolutionary_algorithm(cost_function, matrix, L, K, population_size=100, generations=100, mutation_rate=0.1):
    n = matrix.size(0)
    device = matrix.device
    
    # Initialization
    population = torch.stack([torch.randperm(n)[:L] for _ in range(population_size)], dim=0)
    nearest_indices = get_nearest_indices(matrix, torch.arange(n), K)
    
    for gen in range(generations):
        # Evaluation
        costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
        min_cost_idx = torch.argmin(costs)
        best_solution = population[min_cost_idx]
        
        # Selection
        sorted_indices = torch.argsort(costs)
        selected_indices = sorted_indices[:population_size // 2]
        selected_population = population[selected_indices]
        
        # Crossover
        offspring_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = torch.randperm(population_size)[:2]
            crossover_point = torch.randint(1, L, (1,))
            child1 = torch.cat((selected_population[parent1][:crossover_point], selected_population[parent2][crossover_point:]))
            child2 = torch.cat((selected_population[parent2][:crossover_point], selected_population[parent1][crossover_point:]))
            offspring_population.extend([child1, child2])
        
        # Mutation
        for i in range(len(offspring_population)):
            if torch.rand(1) < mutation_rate:
                mutation_indices = torch.randperm(L)[:2]  # Select 2 random mutation indices
                mutated_sample = offspring_population[i].clone()  # Make a copy of the sample
                
                # Swap the values at the mutation indices
                mutated_sample[mutation_indices[0]], mutated_sample[mutation_indices[1]] = \
                    mutated_sample[mutation_indices[1]], mutated_sample[mutation_indices[0]]
                
                offspring_population[i] = mutated_sample

            if torch.rand(1) < mutation_rate:
                mutation_idx = torch.randint(L, (1,))
                mutated_sample = offspring_population[i].clone()  # Make a copy of the sample
                pool = torch.setdiff1d(torch.arange(n), mutated_sample)
                random_index = torch.randint(len(pool), (1,))
                random_number = pool[random_index]
                mutated_sample[mutation_idx] = random_number.item()
                offspring_population[i] = mutated_sample

        population = torch.stack(offspring_population)
    
    # Final evaluation
    final_costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
    min_cost_idx = torch.argmin(final_costs)
    best_solution = population[min_cost_idx]
    min_cost = final_costs[min_cost_idx]
    
    return best_solution, min_cost