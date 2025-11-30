"""
Consolidate Incremental MMF Results

This script consolidates individual molecule files from incremental saving
into single arrays compatible with the original format.

Usage:
    python consolidate_incremental_results.py --input_dir wavelet_basis/MUTAG --name MUTAG_complete

Author: Khang Nguyen
"""

import torch
import argparse
import json
from pathlib import Path
import glob
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Consolidate incremental MMF results into single arrays'
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing .mol_*.pt files')
    parser.add_argument('--name', type=str, required=True,
                        help='Base name of the experiment')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output name (defaults to input name + "_consolidated")')
    parser.add_argument('--verify', action='store_true',
                        help='Verify completeness before consolidating')
    
    return parser.parse_args()


def find_molecule_files(input_dir, name):
    """
    Find all molecule files for a given experiment.
    
    Returns:
        List of (index, filepath) tuples sorted by index
    """
    pattern = f"{input_dir}/{name}.mol_*.pt"
    files = glob.glob(pattern)
    
    molecule_files = []
    for filepath in files:
        filename = Path(filepath).name
        try:
            idx_str = filename.split('mol_')[1].split('.pt')[0]
            idx = int(idx_str)
            molecule_files.append((idx, filepath))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse index from {filename}")
            continue
    
    # Sort by index
    molecule_files.sort(key=lambda x: x[0])
    
    return molecule_files


def load_metadata(input_dir, name):
    """Load metadata file."""
    metadata_path = Path(input_dir) / f"{name}.metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def consolidate_results(molecule_files, total_molecules, verify=False):
    """
    Consolidate individual molecule files into single arrays.
    
    Args:
        molecule_files: List of (index, filepath) tuples
        total_molecules: Total number of molecules expected
        verify: Whether to verify completeness
        
    Returns:
        Dictionary with consolidated results
    """
    # Initialize arrays
    adjs = [None] * total_molecules
    laplacians = [None] * total_molecules
    mother_coeffs = [None] * total_molecules
    father_coeffs = [None] * total_molecules
    mother_wavelets = [None] * total_molecules
    father_wavelets = [None] * total_molecules
    
    # Track filled indices
    filled_indices = set()
    
    print(f"Consolidating {len(molecule_files)} molecule files...")
    
    for idx, filepath in tqdm(molecule_files, desc="Loading molecules", unit="mol"):
        try:
            result = torch.load(filepath)
            
            # Extract data
            adjs[idx] = result.get('adj')
            laplacians[idx] = result.get('laplacian')
            mother_coeffs[idx] = result.get('mother_coeffs')
            father_coeffs[idx] = result.get('father_coeffs')
            mother_wavelets[idx] = result.get('mother_wavelets')
            father_wavelets[idx] = result.get('father_wavelets')
            
            filled_indices.add(idx)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    # Verify if requested
    if verify:
        expected_indices = set(range(total_molecules))
        missing_indices = expected_indices - filled_indices
        
        if missing_indices:
            print(f"\nWarning: Missing {len(missing_indices)} molecules")
            print(f"Missing indices: {sorted(missing_indices)[:20]}")
            if len(missing_indices) > 20:
                print(f"... and {len(missing_indices) - 20} more")
    
    consolidated = {
        'adjs': adjs,
        'laplacians': laplacians,
        'mother_coeffs': mother_coeffs,
        'father_coeffs': father_coeffs,
        'mother_wavelets': mother_wavelets,
        'father_wavelets': father_wavelets,
        'filled_indices': filled_indices
    }
    
    return consolidated


def save_consolidated_results(consolidated, output_base, metadata):
    """Save consolidated results to files."""
    print("\nSaving consolidated results...")
    
    suffixes = ['adjs', 'laplacians', 'mother_coeffs', 'father_coeffs', 
                'mother_wavelets', 'father_wavelets']
    
    for suffix in suffixes:
        if consolidated[suffix] is not None:
            output_path = f"{output_base}.{suffix}.pt"
            torch.save(consolidated[suffix], output_path)
            print(f"Saved: {output_path}")
    
    # Save metadata
    metadata['consolidated'] = True
    metadata['filled_count'] = len(consolidated['filled_indices'])
    
    metadata_path = f"{output_base}.metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 80)
    print("CONSOLIDATE INCREMENTAL MMF RESULTS")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Experiment name: {args.name}")
    print("-" * 80)
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = load_metadata(args.input_dir, args.name)
    
    if metadata is None:
        print("Warning: No metadata file found")
        total_molecules = None
    else:
        total_molecules = metadata.get('total_molecules')
        print(f"Dataset: {metadata.get('dataset')}")
        print(f"Total molecules: {total_molecules}")
        print(f"Method: {metadata.get('method')}")
    
    # Find all molecule files
    print("\nFinding molecule files...")
    molecule_files = find_molecule_files(args.input_dir, args.name)
    
    if not molecule_files:
        print(f"ERROR: No molecule files found matching pattern {args.name}.mol_*.pt")
        return
    
    print(f"Found {len(molecule_files)} molecule files")
    print(f"Index range: {molecule_files[0][0]} to {molecule_files[-1][0]}")
    
    # Determine total molecules if not in metadata
    if total_molecules is None:
        total_molecules = molecule_files[-1][0] + 1
        print(f"Using inferred total molecules: {total_molecules}")
    
    # Consolidate results
    print("\n" + "-" * 80)
    consolidated = consolidate_results(molecule_files, total_molecules, verify=args.verify)
    
    print(f"\nConsolidation complete:")
    print(f"  Filled: {len(consolidated['filled_indices'])} / {total_molecules} molecules")
    
    # Check completeness
    completeness = len(consolidated['filled_indices']) / total_molecules * 100
    print(f"  Completeness: {completeness:.1f}%")
    
    if completeness < 100:
        print(f"\n⚠  Warning: Dataset is incomplete ({completeness:.1f}%)")
        if args.verify:
            response = input("Continue saving incomplete results? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Determine output name
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"{args.name}_consolidated"
    
    output_base = Path(args.input_dir) / output_name
    
    # Save consolidated results
    print("\n" + "-" * 80)
    if metadata:
        save_consolidated_results(consolidated, output_base, metadata)
    else:
        # Create minimal metadata
        minimal_metadata = {
            'total_molecules': total_molecules,
            'filled_count': len(consolidated['filled_indices']),
            'consolidated': True,
            'source_name': args.name
        }
        save_consolidated_results(consolidated, output_base, minimal_metadata)
    
    print("\n" + "=" * 80)
    print("CONSOLIDATION COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {output_base}.*")
    print(f"Filled: {len(consolidated['filled_indices'])} / {total_molecules} molecules")
    print("=" * 80)


if __name__ == '__main__':
    main()
