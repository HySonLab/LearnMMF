"""
Metaheuristics MMF Basis - Graph Wavelet Basis Generation (Incremental Saving)

This script generates wavelet bases for molecular graphs using metaheuristic
optimization methods with incremental saving after each molecule.

Features:
- Sort molecules by size for efficient batch processing
- Process specific ranges of molecules
- Incremental saving (saves after each molecule to prevent data loss)
- Resume capability (skips already processed molecules)
- Parallel-friendly design for distributed computation

Author: Khang Nguyen
"""

import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import json
import glob

# Add source directory to path
import sys
sys.path.append('../../source/')
from metaheuristics import generate_wavelet_basis
from Dataset import Dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate metaheuristic MMF wavelet basis (incremental saving)'
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
    
    # Processing range arguments
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Start index for processing range (inclusive, after sorting)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index for processing range (exclusive, after sorting)')
    parser.add_argument('--sort_by_size', action='store_true',
                        help='Sort molecules by size (number of atoms) before processing')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing progress (skip already processed molecules)')
    
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


def get_molecule_sizes(data):
    """
    Get the size (number of atoms) of each molecule.
    
    Args:
        data: Dataset object
        
    Returns:
        List of tuples (original_index, size)
    """
    sizes = []
    for idx, molecule in enumerate(data.molecules):
        sizes.append((idx, molecule.nAtoms))
    return sizes


def sort_molecules_by_size(data):
    """
    Sort molecules by size and return mapping.
    
    Args:
        data: Dataset object
        
    Returns:
        Tuple of (sorted_indices, size_info)
        - sorted_indices: List of original indices in sorted order
        - size_info: List of (original_idx, size) tuples in sorted order
    """
    sizes = get_molecule_sizes(data)
    # Sort by size (ascending order - small to large)
    sizes_sorted = sorted(sizes, key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sizes_sorted]
    
    return sorted_indices, sizes_sorted


def get_processed_molecules(output_base):
    """
    Find which molecules have already been processed.
    
    Args:
        output_base: Base path for output files
        
    Returns:
        Set of processed molecule indices
    """
    pattern = f"{output_base}.mol_*.pt"
    processed_files = glob.glob(pattern)
    
    processed_indices = set()
    for filepath in processed_files:
        # Extract index from filename like "output.mol_00042.pt"
        filename = Path(filepath).name
        try:
            idx_str = filename.split('mol_')[1].split('.pt')[0]
            idx = int(idx_str)
            processed_indices.add(idx)
        except (IndexError, ValueError):
            continue
    
    return processed_indices


def save_single_result(output_base, original_idx, adj, laplacian, m_coeffs, f_coeffs, m_wavelets, f_wavelets):
    """
    Save results for a single molecule incrementally.
    
    Args:
        output_base: Base path for output files
        original_idx: Original index of the molecule
        adj, laplacian, m_coeffs, f_coeffs, m_wavelets, f_wavelets: Results to save
    """
    result = {
        'index': original_idx,
        'adj': adj,
        'laplacian': laplacian,
        'mother_coeffs': m_coeffs,
        'father_coeffs': f_coeffs,
        'mother_wavelets': m_wavelets,
        'father_wavelets': f_wavelets
    }
    
    # Save to individual file
    single_file = f"{output_base}.mol_{original_idx:05d}.pt"
    torch.save(result, single_file)


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
    log_file = open(log_path, "a" if args.resume else "w")  # Append if resuming
    
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
    log("=" * 70)
    if args.resume:
        log("RESUMING EXPERIMENT")
    log(f"Experiment: {args.name}")
    log(f"Dataset: {args.dataset}")
    log(f"Output directory: {args.dir}")
    log(f"Device: {args.device}")
    log(f"Metaheuristic method: {method_full_name} ({args.method})")
    log(f"Hyperparameters: K={args.K}, dim={args.dim}")
    log(f"Optimization: epochs={args.epochs}, learning_rate={args.learning_rate}")
    log(f"Random seed: {args.seed}")
    log(f"Incremental saving: ENABLED")
    log("-" * 70)
    
    # Load dataset
    log("Loading dataset...")
    data = load_dataset(args.data_folder, args.dataset)
    num_molecules = data.nMolecules
    log(f"Loaded {num_molecules} molecules")
    
    # Sort molecules by size if requested
    if args.sort_by_size:
        log("Sorting molecules by size...")
        sorted_indices, size_info = sort_molecules_by_size(data)
        log(f"Molecules sorted (size range: {size_info[0][1]} to {size_info[-1][1]} atoms)")
        
        # Save sorting information
        sorting_info = {
            'sorted_indices': sorted_indices,
            'sizes': [(idx, size) for idx, size in size_info]
        }
        sorting_path = output_dir / f"{args.dataset}_sorting_info.json"
        with open(sorting_path, 'w') as f:
            json.dump(sorting_info, f, indent=2)
        log(f"Sorting info saved to: {sorting_path}")
    else:
        sorted_indices = list(range(num_molecules))
        size_info = get_molecule_sizes(data)
        log("Processing molecules in original order")
    
    # Determine processing range
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else num_molecules
    
    # Validate range
    start_idx = max(0, start_idx)
    end_idx = min(num_molecules, end_idx)
    
    if start_idx >= end_idx:
        log(f"ERROR: Invalid range [{start_idx}, {end_idx})")
        log_file.close()
        return
    
    processing_indices = sorted_indices[start_idx:end_idx]
    num_to_process = len(processing_indices)
    
    log(f"Processing range: [{start_idx}, {end_idx}) = {num_to_process} molecules")
    if args.sort_by_size:
        size_range_start = size_info[start_idx][1]
        size_range_end = size_info[end_idx-1][1] if end_idx > 0 else size_range_start
        log(f"Size range: {size_range_start} to {size_range_end} atoms")
    
    # Check for already processed molecules if resuming
    output_base = output_dir / args.name
    processed_molecules = set()
    
    if args.resume:
        log("Checking for already processed molecules...")
        processed_molecules = get_processed_molecules(output_base)
        log(f"Found {len(processed_molecules)} already processed molecules")
        
        # Filter out already processed
        remaining_indices = [idx for idx in processing_indices if idx not in processed_molecules]
        log(f"Remaining to process: {len(remaining_indices)} molecules")
        processing_indices = remaining_indices
        num_to_process = len(processing_indices)
    
    log("-" * 70)
    
    # Process molecules sequentially with incremental saving
    log(f"Processing {num_to_process} molecules...")
    log("Saving after each molecule (incremental mode)")
    log("-" * 70)
    
    successful_count = 0
    skipped_count = 0
    
    for original_idx in tqdm(processing_indices, desc="Processing molecules", unit="mol"):
        molecule_size = data.molecules[original_idx].nAtoms
        log(f"Processing molecule {original_idx} ({molecule_size} atoms)...")
        
        adj, laplacian, m_coeffs, f_coeffs, m_wavelets, f_wavelets = process_single_molecule(
            original_idx, data.molecules[original_idx], 
            args.K, args.dim, args.method, args.epochs, args.learning_rate
        )
        
        # Save immediately after processing
        save_single_result(output_base, original_idx, adj, laplacian, 
                          m_coeffs, f_coeffs, m_wavelets, f_wavelets)
        
        # Count successful vs skipped
        if m_coeffs is not None and f_coeffs is not None:
            successful_count += 1
            log(f"✓ Molecule {original_idx} completed successfully")
        else:
            skipped_count += 1
            log(f"✗ Molecule {original_idx} skipped")
    
    log("-" * 70)
    log(f"Processing complete: {successful_count} successful, {skipped_count} skipped")
    
    # Save metadata
    log("Saving metadata...")
    
    metadata = {
        'dataset': args.dataset,
        'method': args.method,
        'K': args.K,
        'dim': args.dim,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'total_molecules': num_molecules,
        'processed_range': [start_idx, end_idx],
        'processed_count': num_to_process,
        'successful_count': successful_count,
        'skipped_count': skipped_count,
        'sorted_by_size': args.sort_by_size,
        'incremental_saving': True,
        'resumed': args.resume
    }
    
    metadata_path = output_dir / f"{args.name}.metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log(f"Metadata saved to: {metadata_path}")
    log(f"Individual molecule files saved to: {output_base}.mol_*.pt")
    log("=" * 70)
    log("Done!")
    log("")
    log("To consolidate individual files into single arrays, run:")
    log(f"  python consolidate_incremental_results.py --input_dir {output_dir} --name {args.name}")
    log("=" * 70)
    
    log_file.close()


if __name__ == '__main__':
    main()