import torch
from math import sqrt
from learn_batch_mmf import train_learn_batch_mmf


def get_cost(A, L, k, wavelet_indices, rest_indices):
    return train_learn_batch_mmf(A, L, k, wavelet_indices, rest_indices, logging=False)[3]


def generate_random_weighted_graph_laplacian(batch_size, matrix_size, edge_probability, weight_range=(1, 10), device='cpu'):
    # Generate adjacency matrices for the entire batch
    adjacency_matrices = torch.rand(batch_size, matrix_size, matrix_size, device=device) < sqrt(edge_probability)
    adjacency_matrices = torch.minimum(adjacency_matrices, adjacency_matrices.transpose(1, 2))
    adjacency_matrices = adjacency_matrices.to(torch.float32)
    
    # Ensure no self-loops
    mask = ~torch.eye(matrix_size, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    adjacency_matrices = adjacency_matrices * mask
    
    # Generate random weights for the edges
    min_weight, max_weight = weight_range
    weights = torch.randint(min_weight, max_weight + 1, size=(batch_size, matrix_size, matrix_size), dtype=torch.float32, device=device)
    
    # Apply weights to the adjacency matrices
    weighted_adjacency_matrices = adjacency_matrices * weights
    weighted_adjacency_matrices = torch.maximum(weighted_adjacency_matrices, weighted_adjacency_matrices.transpose(1, 2))
    
    # Calculate degree matrices
    degree_matrices = torch.sum(weighted_adjacency_matrices, dim=2)
    degree_matrices = torch.diag_embed(degree_matrices)
    
    # Calculate Laplacian matrices
    laplacian_matrices = degree_matrices - weighted_adjacency_matrices

    # Calculate other node features
    degree = torch.sum(adjacency_matrices, dim=-1)
    weight_sum = torch.sum(weighted_adjacency_matrices, dim=-1)
    node_features = torch.stack([degree, weight_sum], dim=-1)

    return {
        'x': node_features,
        'A': laplacian_matrices
    }