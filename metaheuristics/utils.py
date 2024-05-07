import torch


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