import math
import time
import torch
import random
import numpy as np
from learnable_mmf_model import learnable_mmf_train


def generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability, device='cpu'):
    # Generate adjacency matrix
    adjacency_matrix = torch.rand(matrix_size, matrix_size, device=device) < torch.sqrt(torch.tensor(edge_probability))
    adjacency_matrix = torch.minimum(adjacency_matrix, adjacency_matrix.t())
    adjacency_matrix = adjacency_matrix.to(torch.float32)
    
    # Ensure no self-loops
    adjacency_matrix = adjacency_matrix * (~torch.eye(matrix_size, dtype=torch.bool, device=device))
    
    # Generate random weights for the edges
    weights = torch.rand(size=(matrix_size, matrix_size), dtype=torch.float32, device=device) * 100
    
    # Apply weights to the adjacency matrix
    weighted_adjacency_matrix = adjacency_matrix * weights
    weighted_adjacency_matrix = torch.maximum(weighted_adjacency_matrix, weighted_adjacency_matrix.t())
    
    # Calculate degree matrix
    degree_matrix = torch.sum(weighted_adjacency_matrix, dim=1)
    degree_matrix = torch.diag(degree_matrix)
    
    # Calculate Laplacian matrix
    laplacian_matrix = degree_matrix - weighted_adjacency_matrix

    return laplacian_matrix


def get_nearest_indices(matrix, indices, K):
    N = matrix.size(1)
    L = len(indices)
    
    # Compute the Euclidean distance between each row and the input indices row
    input_rows = matrix[indices]
    distances = torch.cdist(matrix, input_rows) 

    # Exclude the input indices from consideration
    distances[indices, torch.arange(L)] = float('inf')
    
    # Find the K nearest rows for each input index
    nearest_indices = torch.argsort(distances, dim=0)[:K]
    
    return nearest_indices.t()


def get_cost(matrix, indices, rest, L, K, device='cpu'):
    """
    Optimized version of get_cost for CPU execution.
    
    Args:
        matrix: Input matrix (NxN tensor)
        indices: Wavelet indices (L tensor)
        rest: Rest indices (LxK-1 tensor or list)
        L: Number of wavelets
        K: Size of rotation (typically 2)
        device: Device to run on (default 'cpu')
    
    Returns:
        Reconstruction error (Frobenius norm)
    """
    N = matrix.size(0)
    
    # Pre-convert indices to lists once (avoid repeated conversions in loop)
    if indices.dim() == 0:
        wavelet_indices = [indices.item()]
    else:
        wavelet_indices = indices.squeeze().tolist() if indices.dim() > 1 else indices.tolist()
    
    if isinstance(rest, torch.Tensor):
        rest_indices = rest.tolist()
    else:
        rest_indices = rest
    
    # Use float64 for better numerical stability on CPU
    dtype = torch.float64
    A = matrix.to(dtype=dtype, device=device).clone()
    
    # Pre-allocate boolean mask (more efficient than float mask)
    active_index = torch.ones(N, dtype=torch.bool, device=device)
    
    # Pre-allocate identity matrix
    eye_N = torch.eye(N, dtype=dtype, device=device)
    
    # Initialize right as identity
    right = eye_N.clone()
    
    # Pre-allocate U matrix to avoid repeated allocation
    U = torch.empty(N, N, dtype=dtype, device=device)
    
    for l in range(L):
        # Build current indices
        current_indices = [wavelet_indices[l]] + rest_indices[l]
        current_indices.sort()
        
        # Create index tensor for advanced indexing
        idx_tensor = torch.tensor(current_indices, dtype=torch.long, device=device)
        
        # Extract submatrix using index_select (faster than boolean indexing)
        A_rows = torch.index_select(A, 0, idx_tensor)
        A_part = torch.index_select(A_rows, 1, idx_tensor)
        
        # Eigendecomposition
        # For K=2 with symmetric matrices, symeig is faster than eig
        if K == 2:
            values, vectors = torch.symeig(A_part, eigenvectors=True)
        else:
            # Fallback for K != 2 (though K=2 in your case)
            values, vectors = torch.eig(A_part, eigenvectors=True)
        
        # Rotation matrix (transpose)
        O = vectors.t()
        
        # Build U matrix efficiently
        U.copy_(eye_N)  # Start with identity
        # Fill in the rotation block
        for i, idx_i in enumerate(current_indices):
            for j, idx_j in enumerate(current_indices):
                U[idx_i, idx_j] = O[i, j]
        
        # Accumulate right transformation
        # Use torch.mm (faster than matmul on CPU)
        right = torch.mm(U, right)
        
        # Update A = U @ A @ U^T
        temp = torch.mm(U, A)
        A.copy_(torch.mm(temp, U.t()))  # In-place copy
        
        # Mark wavelet as dropped
        active_index[wavelet_indices[l]] = False
    
    # Build left_index matrix efficiently
    active_float = active_index.to(dtype=dtype)
    active_mask = torch.outer(active_float, active_float)
    left_index = eye_N - torch.diag(torch.diag(active_mask)) + active_mask
    
    # Element-wise multiplication for D
    D = A * left_index
    
    # Reconstruction: A_rec = right^T @ D @ right
    temp = torch.mm(right.t(), D)
    A_rec = torch.mm(temp, right)
    
    # Compute Frobenius norm of difference
    diff = matrix.to(dtype=dtype) - A_rec
    return torch.norm(diff, p='fro').item()


def get_cost_numpy(matrix, indices, rest, L, K, device='cpu'):
    """
    Numpy-based version which can be faster for some CPU operations.
    Use this if the torch version isn't fast enough.
    """
    import numpy as np
    
    N = matrix.size(0)
    
    # Convert to numpy
    matrix_np = matrix.cpu().numpy()
    A = matrix_np.copy()
    
    # Pre-convert indices
    if indices.dim() == 0:
        wavelet_indices = [indices.item()]
    else:
        wavelet_indices = indices.squeeze().tolist() if indices.dim() > 1 else indices.tolist()
    
    if isinstance(rest, torch.Tensor):
        rest_indices = rest.tolist()
    else:
        rest_indices = rest
    
    active_index = np.ones(N, dtype=bool)
    eye_N = np.eye(N)
    right = eye_N.copy()
    
    for l in range(L):
        # Build index mask
        current_indices = [wavelet_indices[l]] + rest_indices[l]
        current_indices.sort()
        
        # Boolean mask for indexing
        mask = np.zeros(N, dtype=bool)
        mask[current_indices] = True
        
        # Extract submatrix using fancy indexing
        A_part = A[np.ix_(mask, mask)]
        
        # Eigendecomposition (eigh for symmetric matrices - faster)
        values, vectors = np.linalg.eigh(A_part)
        O = vectors.T
        
        # Build U matrix
        U = eye_N.copy()
        mask_idx = np.where(mask)[0]
        for i, idx_i in enumerate(mask_idx):
            for j, idx_j in enumerate(mask_idx):
                U[idx_i, idx_j] = O[i, j]
        
        # Accumulate right
        right = U @ right
        
        # Update A
        A = U @ A @ U.T
        
        # Drop wavelet
        active_index[wavelet_indices[l]] = False
    
    # Build left_index
    active_mask = active_index[:, None] & active_index[None, :]
    left_index = eye_N.copy()
    left_index[active_mask] = 1.0
    
    # Compute D
    D = A * left_index
    
    # Reconstruction
    A_rec = right.T @ D @ right
    
    # Frobenius norm
    diff = matrix_np - A_rec
    return np.linalg.norm(diff, 'fro')


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
        start_time = time.time()
        # Evaluation
        costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
        min_cost_idx = torch.argmin(costs)
        best_solution = population[min_cost_idx]

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

        end_time = time.time()
        print(f"Generation {gen}: time = {end_time - start_time:.4f} seconds")  

        population = torch.stack(offspring_population)
    
    # Final evaluation
    final_costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
    min_cost_idx = torch.argmin(final_costs)
    best_solution = population[min_cost_idx]
    min_cost = final_costs[min_cost_idx]
    if min_cost < all_time_min_cost:
        all_time_min_cost = min_cost
        all_time_best_solution = best_solution
    
    return all_time_best_solution, nearest_indices[all_time_best_solution, :], all_time_min_cost, min_cost_per_gen, mean_cost_per_gen, all_time_min_cost_per_gen


def directed_evolution(cost_function, matrix, L, K, population_size=100, generations=100, sample_kept_rate=0.3):
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
        start_iter = time.time()
        # Evaluation
        start_time = time.time()
        costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
        end_time = time.time()
        print(f"Generation {gen}: Evaluation time = {end_time - start_time:.4f} seconds")
        min_cost_idx = torch.argmin(costs)
        best_solution = population[min_cost_idx]

        min_cost_per_gen.append(torch.min(costs, dim=0)[0].item())
        mean_cost_per_gen.append(torch.mean(costs, dim=0).item())

        if min_cost_per_gen[-1] < all_time_min_cost:
            all_time_min_cost = min_cost_per_gen[-1]
            all_time_best_solution = best_solution

        all_time_min_cost_per_gen.append(all_time_min_cost)
        
        # Selection
        sorted_indices = torch.argsort(costs)
        selected_indices = sorted_indices[: math.floor(population_size * sample_kept_rate)]
        selected_population = population[selected_indices]

        for s in selected_population:
            assert len(s.tolist()) == len(set(s.tolist()))
        
        # Diversification
        offspring_population = []
        for i in range(len(selected_population)):
            for _ in range(population_size // len(selected_population) + 1):
                mutation_indices = torch.randperm(L)[:2]  # Select 2 random mutation indices
                mutated_sample = selected_population[i].clone()  # Make a copy of the sample
                # Swap the values at the mutation indices
                mutated_sample[mutation_indices[0]], mutated_sample[mutation_indices[1]] = \
                    mutated_sample[mutation_indices[1]].item(), mutated_sample[mutation_indices[0]].item()
            
                mutation_idx = torch.randint(L, (1,))
                pool = torch.from_numpy(np.setdiff1d(torch.arange(n).numpy(), mutated_sample.numpy()))
                random_index = torch.randint(len(pool), (1,))
                random_number = pool[random_index]
                mutated_sample[mutation_idx] = random_number.item()
                offspring_population.append(mutated_sample)

        random.shuffle(offspring_population)
        offspring_population = offspring_population[: population_size - len(selected_population)]
        population = torch.stack(offspring_population + [sample for sample in selected_population])
        assert len(population) == population_size
        end_time = time.time()
        print(f"Generation {gen}: time = {end_time - start_iter:.4f} seconds")
    
    # Final evaluation
    final_costs = torch.tensor([cost_function(matrix, indices, nearest_indices[indices, :], L, K) for indices in population], device=device)
    min_cost_idx = torch.argmin(final_costs)
    best_solution = population[min_cost_idx]
    min_cost = final_costs[min_cost_idx]
    if min_cost < all_time_min_cost:
        all_time_min_cost = min_cost
        all_time_best_solution = best_solution
    
    return all_time_best_solution, nearest_indices[all_time_best_solution, :], all_time_min_cost, min_cost_per_gen, mean_cost_per_gen, all_time_min_cost_per_gen


def generate_wavelet_basis(matrix, L, K, method, epochs=1024, learning_rate=1e-3):
    wavelet_indices, rest_indices = None, None
    if method == 'evolutionary_algorithm':
        wavelet_indices, rest_indices, _, _, _, _ = evolutionary_algorithm(get_cost_numpy, matrix, L = L, K = K, population_size = 20, generations = 40, mutation_rate = 0.2)
    elif method == 'directed_evolution':
        wavelet_indices, rest_indices, _, _, _, _ = directed_evolution(get_cost, matrix, L = L, K = K, population_size = 10, generations = 40, sample_kept_rate = 0.3)
    dim = matrix.size(0) - L
    return learnable_mmf_train(matrix, L = L, K = K, drop = 1, dim = dim, wavelet_indices = wavelet_indices.unsqueeze(-1).tolist(), rest_indices = rest_indices.tolist(), epochs = epochs, learning_rate = learning_rate, early_stop = True)