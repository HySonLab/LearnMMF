"""
Metaheuristics MMF Basis - Graph Wavelet Basis Generation

This script generates wavelet bases for molecular graphs using metaheuristic
optimization methods (Evolutionary Algorithm or Directed Evolution).

Author: Khang Nguyen
"""

import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

# Add source directory to path
import sys
sys.path.append('../../source/')
from metaheuristics import generate_wavelet_basis
from Dataset import Dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate metaheuristic MMF wavelet basis'
    )
    
    # I/O arguments
    parser.add_argument('--dir', type=str, default='wavelet_basis/',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='NAME',
                        help='Experiment name')
    parser.add_argument('--dataset', type=str, default='.',
                        help='Graph kernel benchmark dataset')
    parser.add_argument('--data_folder', type=str, default='../../data/',
                        help='Data folder path')
    
    # Model hyperparameters
    parser.add_argument('--K', type=int, default=2,
                        help='Size of the rotation matrix')
    parser.add_argument('--dim', type=int, default=2,
                        help='Dimension left at the end')
    
    # Metaheuristic method
    parser.add_argument('--method', type=str, default='ea',
                        choices=['ea', 'de'],
                        help='Metaheuristic method: ea (Evolutionary Algorithm) or de (Directed Evolution)')
    parser.add_argument('--epochs', type=int, default=1024,
                        help='Number of epochs for optimization')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimization')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=123456789,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_normalized_laplacian(adj_matrix):
    """
    Compute normalized graph Laplacian from adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix (N x N)
        
    Returns:
        Normalized Laplacian matrix
    """
    N = adj_matrix.size(0)
    
    # Add self-loops
    adj_with_self_loops = adj_matrix + torch.eye(N)
    
    # Compute degree matrix
    degrees = torch.sum(adj_with_self_loops, dim=0)
    degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))
    
    # Normalized Laplacian: D^(-1/2) * (D - A) * D^(-1/2)
    degree_matrix = torch.diag(degrees)
    laplacian_norm = torch.matmul(
        torch.matmul(degree_inv_sqrt, degree_matrix - adj_with_self_loops),
        degree_inv_sqrt
    )
    
    return laplacian_norm


def process_single_molecule(sample_idx, molecule, K, dim, method, epochs, learning_rate):
    """
    Process a single molecule to compute wavelet basis using metaheuristics.
    
    Args:
        sample_idx: Index of the sample
        molecule: Molecule object
        K: Size of rotation matrix
        dim: Dimension left at the end
        method: Metaheuristic method ('ea' or 'de')
        epochs: Number of optimization epochs
        learning_rate: Learning rate for optimization
    
    Returns:
        Tuple containing (adj, laplacian, mother_coeffs, father_coeffs, 
                         mother_wavelets, father_wavelets)
    """
    N = molecule.nAtoms
    
    # Skip molecules that are too large
    if N > 512:
        return (None, None, None, None, None, None)
    
    # Build adjacency matrix
    adj = torch.zeros(N, N)
    for v in range(N):
        neighbors = molecule.atoms[v].neighbors
        adj[v, neighbors] = 1
    
    # Compute normalized graph Laplacian
    laplacian = compute_normalized_laplacian(adj)
    
    # Calculate number of wavelet levels
    L = N - dim
    
    if L > 0:
        try:
            # Map method abbreviation to full name
            method_name = {
                'ea': 'evolutionary_algorithm',
                'de': 'directed_evolution'
            }.get(method, 'evolutionary_algorithm')
            
            # Generate wavelet basis using metaheuristic
            A_rec, U, D, mother_coefficients, father_coefficients, \
                mother_wavelets_raw, father_wavelets_raw = generate_wavelet_basis(
                    laplacian,
                    L=L,
                    K=K,
                    method=method_name,
                    epochs=epochs,
                    learning_rate=learning_rate,
                )
                        
            # Sort and extract mother coefficients
            mother_diag = torch.diag(mother_coefficients).unsqueeze(dim=0)
            mother_coeffs_sorted, _ = torch.sort(mother_diag, descending=True)
            
            # Sort and extract father coefficients
            father_coeffs_sorted, _ = torch.sort(
                father_coefficients.flatten(),
                descending=True
            )
            father_coeffs_sorted = father_coeffs_sorted.unsqueeze(dim=0)
            
            # Prepare wavelets for storage
            mother_wavelets_out = mother_wavelets_raw.unsqueeze(dim=0)
            father_wavelets_out = father_wavelets_raw.unsqueeze(dim=0)
            
            # Detach all tensors to remove gradient tracking
            return (
                adj.detach(),
                laplacian.detach(),
                mother_coeffs_sorted.detach(),
                father_coeffs_sorted.detach(),
                mother_wavelets_out.detach(),
                father_wavelets_out.detach()
            )
            
        except Exception as e:
            print(f"Error processing molecule {sample_idx}: {e}")
            return (
                adj.detach() if adj is not None else None,
                laplacian.detach() if laplacian is not None else None,
                None, None, None, None
            )
    else:
        # Invalid molecule (L <= 0)
        return (
            adj.detach(),
            laplacian.detach(),
            None, None, None, None
        )


def load_dataset(data_folder, dataset_name):
    """Load dataset from files."""
    data_fn = f'{data_folder}/{dataset_name}/{dataset_name}.dat'
    meta_fn = f'{data_folder}/{dataset_name}/{dataset_name}.meta'
    return Dataset(data_fn, meta_fn)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = output_dir / f"{args.name}.log"
    log_file = open(log_path, "w")
    
    def log(message):
        """Helper function to log messages."""
        print(message)
        log_file.write(message + "\n")
        log_file.flush()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Map method abbreviation to full name for display
    method_full_name = {
        'ea': 'Evolutionary Algorithm',
        'de': 'Directed Evolution'
    }.get(args.method, args.method)
    
    # Log configuration
    log(f"Experiment: {args.name}")
    log(f"Dataset: {args.dataset}")
    log(f"Output directory: {args.dir}")
    log(f"Device: {args.device}")
    log(f"Metaheuristic method: {method_full_name} ({args.method})")
    log(f"Hyperparameters: K={args.K}, dim={args.dim}")
    log(f"Optimization: epochs={args.epochs}, learning_rate={args.learning_rate}")
    log(f"Random seed: {args.seed}")
    log("-" * 60)
    
    # Load dataset
    log("Loading dataset...")
    data = load_dataset(args.data_folder, args.dataset)
    num_molecules = data.nMolecules
    log(f"Loaded {num_molecules} molecules")
    log("-" * 60)
    
    # Initialize result containers
    adjs = []
    laplacians = []
    mother_coeffs = []
    father_coeffs = []
    mother_wavelets = []
    father_wavelets = []
    
    # Process molecules sequentially
    log(f"Processing {num_molecules} molecules...")
    log("-" * 60)
    
    successful_count = 0
    skipped_count = 0
    
    for idx in tqdm(range(num_molecules), desc="Processing molecules", unit="mol"):
        adj, laplacian, m_coeffs, f_coeffs, m_wavelets, f_wavelets = process_single_molecule(
            idx, data.molecules[idx], args.K, args.dim, args.method, args.epochs, args.learning_rate
        )
        
        adjs.append(adj)
        laplacians.append(laplacian)
        mother_coeffs.append(m_coeffs)
        father_coeffs.append(f_coeffs)
        mother_wavelets.append(m_wavelets)
        father_wavelets.append(f_wavelets)
        
        # Count successful vs skipped
        if m_coeffs is not None and f_coeffs is not None:
            successful_count += 1
        else:
            skipped_count += 1
    
    log("-" * 60)
    log(f"Processing complete: {successful_count} successful, {skipped_count} skipped")
    
    # Verify data integrity
    assert len(adjs) == num_molecules
    assert len(laplacians) == num_molecules
    assert len(mother_coeffs) == num_molecules
    assert len(father_coeffs) == num_molecules
    assert len(mother_wavelets) == num_molecules
    assert len(father_wavelets) == num_molecules
    
    # Save results
    log("Saving results...")
    output_base = output_dir / args.name
    
    torch.save(adjs, f"{output_base}.adjs.pt")
    torch.save(laplacians, f"{output_base}.laplacians.pt")
    torch.save(mother_coeffs, f"{output_base}.mother_coeffs.pt")
    torch.save(father_coeffs, f"{output_base}.father_coeffs.pt")
    torch.save(mother_wavelets, f"{output_base}.mother_wavelets.pt")
    torch.save(father_wavelets, f"{output_base}.father_wavelets.pt")
    
    log(f"Results saved to: {output_base}.*")
    log("=" * 60)
    log("Done!")
    
    log_file.close()


if __name__ == '__main__':
    main()