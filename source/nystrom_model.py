import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
import time

# Nystrom method
def nystrom_model(A, dim):
    N = A.size(0)
    perm = torch.randperm(N).detach().cpu().numpy()
    perm = perm[:dim]
    index = np.zeros(N)
    for k in range(perm.shape[0]):
        index[perm[k]] = 1
    index = torch.Tensor(index)
    outer = torch.outer(index, index)

    C = torch.transpose(A[index == 1], 0, 1)
    W = torch.reshape(A[outer == 1], (dim, dim))
    W_inverse = torch.matmul(torch.inverse(torch.matmul(torch.transpose(W, 0, 1), W) + 1e-2 * torch.eye(dim)), torch.transpose(W, 0, 1))
    A_rec = torch.matmul(torch.matmul(C, W_inverse), torch.transpose(C, 0, 1))
    norm = torch.norm(A - A_rec, p = 'fro')
    print('Error = ', norm)

    return A_rec, C, W_inverse


def nystrom_fps_model(A, dim):
    N = A.size(0)
    
    # Greedy Furthest Point Sampling
    selected_indices = []
    remaining = set(range(N))
    
    # Start with a random point
    first_idx = torch.randint(0, N, (1,)).item()
    selected_indices.append(first_idx)
    remaining.remove(first_idx)
    
    # Greedy selection: pick point that maximizes minimum distance to selected set
    for _ in range(dim - 1):
        max_min_dist = -1
        best_idx = None
        
        for candidate in remaining:
            # Compute minimum distance from candidate to all selected points
            min_dist = float('inf')
            for selected in selected_indices:
                # Use matrix entry as distance/dissimilarity measure
                dist = torch.norm(A[candidate] - A[selected])
                min_dist = min(min_dist, dist.item())
            
            # Keep track of candidate with maximum minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = candidate
        
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
    
    # Convert to numpy array for compatibility with original code
    perm = np.array(selected_indices)
    
    # Create index mask (rest of the code stays the same)
    index = np.zeros(N)
    for k in range(perm.shape[0]):
        index[perm[k]] = 1
    index = torch.Tensor(index)
    outer = torch.outer(index, index)

    C = torch.transpose(A[index == 1], 0, 1)
    W = torch.reshape(A[outer == 1], (dim, dim))
    W_inverse = torch.matmul(
        torch.inverse(torch.matmul(torch.transpose(W, 0, 1), W) + 1e-2 * torch.eye(dim)), 
        torch.transpose(W, 0, 1)
    )
    A_rec = torch.matmul(torch.matmul(C, W_inverse), torch.transpose(C, 0, 1))
    norm = torch.norm(A - A_rec, p='fro')
    print('Error = ', norm)

    return A_rec, C, W_inverse