import torch
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

# Baseline MMF
import sys
sys.path.append('../../source/')
from metaheuristics import generate_wavelet_basis

# Data loader
from Dataset import *

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Metaheuristic MMF with Parallelization')
    parser.add_argument('--dir', '-dir', type = str, default = 'wavelet_basis/', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-dataset', type = str, default = '.', help = 'Graph kernel benchmark dataset')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = '../../data/', help = 'Data folder')
    parser.add_argument('--method', '-method', type = str, default = 'evolutionary_algorithm', help = 'Method to select indices')
    parser.add_argument('--K', '-K', type = int, default = 2, help = 'Size of the rotation matrix')
    parser.add_argument('--dim', '-dim', type = int, default = 2, help = 'Dimension left at the end')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    parser.add_argument('--num_workers', '-num_workers', type = int, default = 10, help = 'Number of parallel workers')
    args = parser.parse_args()
    return args


def process_single_molecule(args_tuple):
    """
    Process a single molecule to compute wavelet basis.
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        args_tuple: Tuple containing (sample_idx, molecule, K, dim, seed)
    
    Returns:
        Tuple containing (sample_idx, adj, laplacian, mother_coeffs, father_coeffs, 
                         mother_wavelets, father_wavelets)
    """
    sample_idx, molecule, K, dim, seed = args_tuple
    
    # Set random seeds for this process
    np.random.seed(seed + sample_idx)
    torch.manual_seed(seed + sample_idx)
    
    N = molecule.nAtoms
    
    # Skip molecules that are too large
    if N > 512:
        return (sample_idx, None, None, None, None, None, None)
    
    # Build adjacency matrix
    adj = torch.zeros(N, N)
    for v in range(N):
        neighbors = molecule.atoms[v].neighbors
        adj[v, neighbors] = 1
    
    # Compute normalized graph Laplacian
    adj_self = torch.Tensor(adj) + torch.eye(N)
    D = torch.sum(adj_self, dim=0)
    DD = torch.diag(1.0 / torch.sqrt(D))
    L_norm = torch.matmul(torch.matmul(DD, torch.diag(D) - adj_self), DD)
    
    A = L_norm
    L = N - dim
    
    if L > 0:
        try:
            A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets_, father_wavelets_ = \
                generate_wavelet_basis(A, L=L, K=K, method='evolutionary_algorithm')
            
            # Process mother coefficients
            diag = torch.diag(mother_coefficients).unsqueeze(dim=0)
            values_mother, _ = torch.sort(diag, descending=True)
            
            # Process father coefficients
            values_father, _ = torch.sort(father_coefficients.flatten(), descending=True)
            values_father = values_father.unsqueeze(dim=0)
            
            # Store wavelets
            mother_wavelets_out = mother_wavelets_.unsqueeze(dim=0)
            father_wavelets_out = father_wavelets_.unsqueeze(dim=0)
            
            # Detach all tensors to remove gradient tracking before serialization
            return (sample_idx, 
                   adj.detach(), 
                   L_norm.detach(), 
                   values_mother.detach(), 
                   values_father.detach(), 
                   mother_wavelets_out.detach(), 
                   father_wavelets_out.detach())
        except Exception as e:
            print(f"Error processing molecule {sample_idx}: {e}")
            return (sample_idx, 
                   adj.detach() if adj is not None else None, 
                   L_norm.detach() if L_norm is not None else None, 
                   None, None, None, None)
    else:
        return (sample_idx, 
               adj.detach(), 
               L_norm.detach(), 
               None, None, None, None)


def main():
    args = _parse_args()
    log_name = args.dir + "/" + args.name + ".log"
    LOG = open(log_name, "w")
    
    # Fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device
    print(f"Device: {device}")
    print(f"Name: {args.name}")
    print(f"Directory: {args.dir}")
    print(f"Number of workers: {args.num_workers}")
    
    # Load data
    data_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.dat'
    meta_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.meta'
    
    data = Dataset(data_fn, meta_fn)
    num_data = data.nMolecules
    
    print(f"Total molecules to process: {num_data}")
    
    # Prepare arguments for parallel processing
    process_args = [
        (idx, data.molecules[idx], args.K, args.dim, args.seed)
        for idx in range(num_data)
    ]
    
    # Initialize result containers
    adjs = [None] * num_data
    laplacians = [None] * num_data
    mother_coeffs = [None] * num_data
    father_coeffs = [None] * num_data
    mother_wavelets = [None] * num_data
    father_wavelets = [None] * num_data
    
    # Process molecules in parallel
    print(f"\nProcessing {num_data} molecules using {args.num_workers} workers...")
    
    with Pool(processes=args.num_workers) as pool:
        # Use imap_unordered for better performance with progress bar
        results = list(tqdm(
            pool.imap_unordered(process_single_molecule, process_args),
            total=num_data,
            desc="Processing molecules"
        ))
    
    # Collect results in the correct order
    print("\nCollecting results...")
    for result in results:
        sample_idx, adj, laplacian, m_coeffs, f_coeffs, m_wavelets, f_wavelets = result
        adjs[sample_idx] = adj
        laplacians[sample_idx] = laplacian
        mother_coeffs[sample_idx] = m_coeffs
        father_coeffs[sample_idx] = f_coeffs
        mother_wavelets[sample_idx] = m_wavelets
        father_wavelets[sample_idx] = f_wavelets
    
    # Verify all data is present
    assert len(adjs) == num_data
    assert len(laplacians) == num_data
    assert len(mother_coeffs) == num_data
    assert len(father_coeffs) == num_data
    assert len(mother_wavelets) == num_data
    assert len(father_wavelets) == num_data
    
    # Create output directory if it doesn't exist
    output_dir = args.dir + '/' + args.dataset
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    print("\nSaving results...")
    torch.save(adjs, output_dir + '/' + args.name + '.adjs.pt')
    torch.save(laplacians, output_dir + '/' + args.name + '.laplacians.pt')
    torch.save(mother_coeffs, output_dir + '/' + args.name + '.mother_coeffs.pt')
    torch.save(father_coeffs, output_dir + '/' + args.name + '.father_coeffs.pt')
    torch.save(mother_wavelets, output_dir + '/' + args.name + '.mother_wavelets.pt')
    torch.save(father_wavelets, output_dir + '/' + args.name + '.father_wavelets.pt')
    
    print('Done!')
    LOG.write('Processing completed successfully\n')
    LOG.close()


if __name__ == '__main__':
    main()