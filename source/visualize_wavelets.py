import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Add source directory
sys.path.append('../experiments/graph_classification/')
from baseline_mmf_model import Baseline_MMF
from Dataset import Dataset


def draw_graph_with_wavelet(A, wavelet, file_name, 
                             nodes_x=None, nodes_y=None, edges=None,
                             title=None, figsize=10, dpi=300):
    """
    Draw an arbitrary graph with wavelet visualization.
    
    Parameters:
    -----------
    A : torch.Tensor or numpy.ndarray
        Adjacency matrix of shape (N, N)
    wavelet : torch.Tensor or numpy.ndarray
        Wavelet coefficients of shape (K, N) or (N,)
    file_name : str
        Output file name (without extension)
    nodes_x : list or numpy.ndarray, optional
        X coordinates for nodes. If None, uses circular layout
    nodes_y : list or numpy.ndarray, optional
        Y coordinates for nodes. If None, uses circular layout
    edges : list of tuples, optional
        List of (u, v) edge pairs. If None, derives from adjacency matrix
    title : str, optional
        Plot title
    figsize : int or tuple, optional
        Figure size (default: 10)
    dpi : int, optional
        Resolution for saving (default: 300)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Convert to numpy if needed
    if torch.is_tensor(A):
        A = A.detach().cpu().numpy()
    if torch.is_tensor(wavelet):
        wavelet = wavelet.detach().cpu().numpy()
    
    # Handle wavelet dimensions
    if wavelet.ndim == 1:
        wavelet = wavelet.reshape(1, -1)
    
    N = A.shape[0]
    
    # Generate default circular layout if coordinates not provided
    if nodes_x is None or nodes_y is None:
        R = 10
        nodes_x = []
        nodes_y = []
        for k in range(N):
            alpha = 2 * np.pi * k / N
            x = R * np.cos(alpha)
            y = R * np.sin(alpha)
            nodes_x.append(x)
            nodes_y.append(y)
        nodes_x = np.array(nodes_x)
        nodes_y = np.array(nodes_y)
    
    # Generate edges from adjacency matrix if not provided
    if edges is None:
        edges = []
        for i in range(N):
            for j in range(i + 1, N):  # Only upper triangle to avoid duplicates
                if A[i, j] != 0:
                    edges.append((i, j))
    
    # Initialize z-coordinates (all at zero)
    nodes_z = np.zeros(N)
    
    # Create figure
    plt.clf()
    if isinstance(figsize, (int, float)):
        figsize = (figsize, figsize)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    ax.scatter(nodes_x, nodes_y, nodes_z, c='black', s=50, alpha=0.6)
    
    # Plot edges
    for edge in edges:
        u, v = edge[0], edge[1]
        x1, y1, z1 = nodes_x[u], nodes_y[u], nodes_z[u]
        x2, y2, z2 = nodes_x[v], nodes_y[v], nodes_z[v]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='b', alpha=0.4, linewidth=0.8)
    
    # Plot wavelets as vertical bars
    for k in range(N):
        w = np.sum(wavelet[:, k])
        x1, y1, z1 = nodes_x[k], nodes_y[k], nodes_z[k]
        x2, y2, z2 = x1, y1, z1 + w
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='r', linewidth=2, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wavelet Amplitude')
    if title:
        ax.set_title(title)
    
    # Save figure
    plt.savefig(f'{file_name}.pdf', dpi=dpi, bbox_inches='tight')
    
    return fig


def draw_multiple_wavelets(A, wavelets, output_prefix, 
                           nodes_x=None, nodes_y=None, edges=None,
                           title_prefix='Wavelet', figsize=10, dpi=300):
    """
    Draw multiple wavelets for the same graph.
    
    Parameters:
    -----------
    A : torch.Tensor or numpy.ndarray
        Adjacency matrix of shape (N, N)
    wavelets : list or torch.Tensor or numpy.ndarray
        List of wavelets or tensor of shape (L, K, N)
    output_prefix : str
        Prefix for output file names
    nodes_x, nodes_y, edges : optional
        Graph layout parameters (see draw_graph_with_wavelet)
    title_prefix : str, optional
        Prefix for plot titles
    figsize, dpi : optional
        Figure parameters
    
    Returns:
    --------
    figures : list
        List of created figure objects
    """
    # Convert to list if tensor
    if torch.is_tensor(wavelets):
        wavelets = [wavelets[i] for i in range(wavelets.shape[0])]
    elif isinstance(wavelets, np.ndarray):
        wavelets = [wavelets[i] for i in range(wavelets.shape[0])]
    
    figures = []
    for l, wavelet in enumerate(wavelets):
        file_name = f'{output_prefix}_{l}'
        title = f'{title_prefix} {l}'
        fig = draw_graph_with_wavelet(
            A, wavelet, file_name,
            nodes_x=nodes_x, nodes_y=nodes_y, edges=edges,
            title=title, figsize=figsize, dpi=dpi
        )
        figures.append(fig)
        plt.close(fig)
    
    return figures


def get_graph_layout(A, layout='circular', scale=10, seed=42, **kwargs):
    """
    Generate node coordinates for different graph layouts.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Adjacency matrix
    layout : str
        Layout type: 'circular', 'spring', 'random', 'kamada_kawai', 
        'spectral', 'shell', 'spiral', 'planar', 'fruchterman_reingold'
    scale : float
        Scaling factor for coordinates
    seed : int
        Random seed for reproducibility
    **kwargs : dict
        Additional arguments passed to networkx layout functions
    
    Returns:
    --------
    nodes_x, nodes_y : numpy.ndarray
        Node coordinates
    
    Notes:
    ------
    Layouts requiring networkx:
    - 'kamada_kawai': Force-directed layout using path-length cost function (best for most graphs)
    - 'fruchterman_reingold': Force-directed layout (good alternative to spring)
    - 'spectral': Based on graph Laplacian eigenvectors
    - 'shell': Nodes in concentric circles
    - 'spiral': Nodes arranged in a spiral
    - 'planar': For planar graphs only
    
    If networkx is not installed, falls back to simple implementations.
    """
    N = A.shape[0]
    
    if layout == 'circular':
        nodes_x = []
        nodes_y = []
        for k in range(N):
            alpha = 2 * np.pi * k / N
            x = scale * np.cos(alpha)
            y = scale * np.sin(alpha)
            nodes_x.append(x)
            nodes_y.append(y)
        return np.array(nodes_x), np.array(nodes_y)
    
    elif layout == 'spring':
        # Simple spring layout implementation
        np.random.seed(seed)
        pos = np.random.randn(N, 2) * scale
        
        # Simple force-directed iterations
        for _ in range(50):
            forces = np.zeros((N, 2))
            
            # Repulsive forces between all nodes
            for i in range(N):
                for j in range(N):
                    if i != j:
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff) + 1e-6
                        forces[i] += diff / (dist ** 2)
            
            # Attractive forces for edges
            for i in range(N):
                for j in range(N):
                    if A[i, j] != 0:
                        diff = pos[j] - pos[i]
                        forces[i] += diff * 0.01
            
            pos += forces * 0.1
        
        return pos[:, 0], pos[:, 1]
    
    elif layout == 'random':
        np.random.seed(seed)
        nodes_x = np.random.uniform(-scale, scale, N)
        nodes_y = np.random.uniform(-scale, scale, N)
        return nodes_x, nodes_y
    
    # NetworkX-based layouts
    elif layout in ['kamada_kawai', 'fruchterman_reingold', 'spectral', 
                    'shell', 'spiral', 'planar']:
        try:
            import networkx as nx
        except ImportError:
            print(f"Warning: networkx not installed. Falling back to spring layout.")
            print("Install with: pip install networkx")
            return get_graph_layout(A, layout='spring', scale=scale, seed=seed)
        
        # Create networkx graph from adjacency matrix
        G = nx.from_numpy_array(A)
        
        # Select layout algorithm
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G, scale=scale, **kwargs)
        elif layout == 'fruchterman_reingold':
            pos = nx.spring_layout(G, k=scale/10, iterations=50, seed=seed, **kwargs)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G, scale=scale, **kwargs)
        elif layout == 'shell':
            pos = nx.shell_layout(G, scale=scale, **kwargs)
        elif layout == 'spiral':
            pos = nx.spiral_layout(G, scale=scale, **kwargs)
        elif layout == 'planar':
            if nx.is_planar(G):
                pos = nx.planar_layout(G, scale=scale, **kwargs)
            else:
                print(f"Warning: Graph is not planar. Falling back to kamada_kawai layout.")
                pos = nx.kamada_kawai_layout(G, scale=scale, **kwargs)
        
        # Extract coordinates
        nodes_x = np.array([pos[i][0] for i in range(N)])
        nodes_y = np.array([pos[i][1] for i in range(N)])
        
        return nodes_x, nodes_y
    
    else:
        raise ValueError(f"Unknown layout type: {layout}. Available layouts: "
                        f"'circular', 'spring', 'random', 'kamada_kawai', "
                        f"'fruchterman_reingold', 'spectral', 'shell', 'spiral', 'planar'")


# Example usage function
def example_usage():
    """
    Example demonstrating how to use the graph plotting functions.
    """
    # Create a simple graph (cycle)
    N = 20
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = 1
        A[(i + 1) % N, i] = 1
    
    # Create a simple wavelet
    wavelet = np.sin(np.arange(N) * 2 * np.pi / N).reshape(1, -1)
    
    # Example 1: Default circular layout
    print("Creating circular layout...")
    draw_graph_with_wavelet(A, wavelet, 'example_circular', 
                           title='Cycle Graph - Circular Layout')
    
    # Example 2: Kamada-Kawai layout (best for general graphs)
    print("Creating Kamada-Kawai layout...")
    nodes_x, nodes_y = get_graph_layout(A, layout='kamada_kawai', scale=10)
    draw_graph_with_wavelet(A, wavelet, 'example_kamada_kawai',
                           nodes_x=nodes_x, nodes_y=nodes_y,
                           title='Cycle Graph - Kamada-Kawai Layout')
    
    # Example 3: Fruchterman-Reingold layout (force-directed)
    print("Creating Fruchterman-Reingold layout...")
    nodes_x, nodes_y = get_graph_layout(A, layout='fruchterman_reingold', scale=10)
    draw_graph_with_wavelet(A, wavelet, 'example_fruchterman',
                           nodes_x=nodes_x, nodes_y=nodes_y,
                           title='Cycle Graph - Fruchterman-Reingold Layout')
    
    # Example 4: Spectral layout
    print("Creating spectral layout...")
    nodes_x, nodes_y = get_graph_layout(A, layout='spectral', scale=10)
    draw_graph_with_wavelet(A, wavelet, 'example_spectral',
                           nodes_x=nodes_x, nodes_y=nodes_y,
                           title='Cycle Graph - Spectral Layout')
    
    # Example 5: Multiple wavelets with different layouts
    print("Creating multiple wavelets...")
    wavelets = [np.sin((i + 1) * np.arange(N) * 2 * np.pi / N).reshape(1, -1) 
                for i in range(3)]
    
    # Using Kamada-Kawai for better visualization
    nodes_x, nodes_y = get_graph_layout(A, layout='kamada_kawai', scale=10)
    draw_multiple_wavelets(A, wavelets, 'example_multi_kamada',
                          nodes_x=nodes_x, nodes_y=nodes_y,
                          title_prefix='Harmonic (Kamada-Kawai)')
    
    print("\nAll examples created successfully!")
    print("\nLayout recommendations:")
    print("- kamada_kawai: Best for most graphs (aesthetic, well-spaced)")
    print("- fruchterman_reingold: Good alternative, faster than kamada_kawai")
    print("- spectral: Fast, good for graphs with clear structure")
    print("- circular: Perfect for cycle graphs and symmetric structures")
    print("- shell: Good for graphs with natural groupings")



def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize molecular graphs with MMF wavelets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='Dataset name (e.g., MUTAG, PTC_MR, PROTEINS)')
    parser.add_argument('--data-folder', type=str, default='../data/',
                        help='Path to data folder')
    parser.add_argument('--molecule-id', type=int, default=0,
                        help='Index of molecule to visualize (0-based)')
    parser.add_argument('--molecule-range', type=str, default=None,
                        help='Range of molecules to process (e.g., "0-10" or "5,7,9")')
    
    # MMF parameters
    parser.add_argument('--dim', type=int, default=2,
                        help='Dimension parameter for MMF (number of father wavelets)')
    parser.add_argument('--custom-L', type=int, default=None,
                        help='Custom value for L (number of mother wavelets). If None, uses N-dim')
    
    # Visualization parameters
    parser.add_argument('--layout', type=str, default='kamada_kawai',
                        choices=['circular', 'kamada_kawai', 'fruchterman_reingold', 
                                'spectral', 'shell', 'spiral', 'planar', 'spring', 'random'],
                        help='Graph layout algorithm')
    parser.add_argument('--scale', type=float, default=10.0,
                        help='Scaling factor for node layout')
    parser.add_argument('--figsize', type=int, default=10,
                        help='Figure size in inches')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible layouts')
    
    # Output parameters
    parser.add_argument('--output-folder', type=str, default='./output',
                        help='Folder to save visualization outputs')
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output files (default: dataset_mol_id)')
    parser.add_argument('--save-wavelets', type=str, default='mother',
                        choices=['mother', 'father', 'both', 'none'],
                        help='Which wavelets to save')
    parser.add_argument('--wavelet-range', type=str, default=None,
                        help='Range of wavelets to save (e.g., "0-5" saves first 6). Default: all')
    
    # Display parameters
    parser.add_argument('--show-stats', action='store_true',
                        help='Print detailed statistics')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--use-laplacian', action='store_true', default=True,
                        help='Use normalized Laplacian instead of adjacency matrix')
    parser.add_argument('--no-laplacian', dest='use_laplacian', action='store_false',
                        help='Use adjacency matrix instead of Laplacian')
    
    return parser.parse_args()


def parse_range(range_str):
    """
    Parse range string like "0-10" or "5,7,9" into list of integers.
    
    Examples:
        "0-10" -> [0, 1, 2, ..., 10]
        "5,7,9" -> [5, 7, 9]
        "0-5,10,15-17" -> [0, 1, 2, 3, 4, 5, 10, 15, 16, 17]
    """
    if range_str is None:
        return None
    
    indices = []
    parts = range_str.split(',')
    
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    
    return sorted(set(indices))


def process_molecule(molecule_id, data, args):
    """Process and visualize a single molecule."""
    import os
    
    # Get molecule
    if molecule_id >= data.nMolecules:
        print(f"Warning: Molecule ID {molecule_id} out of range (max: {data.nMolecules-1}). Skipping.")
        return
    
    molecule = data.molecules[molecule_id]
    N = molecule.nAtoms
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"Processing Molecule #{molecule_id}")
        print(f"{'='*60}")
        print(f"  Atoms: {N}")
    
    # Build adjacency matrix
    adj = torch.zeros(N, N)
    for v in range(N):
        adj[v, molecule.atoms[v].neighbors] = 1
    
    # Choose matrix to decompose
    if args.use_laplacian:
        # Compute normalized Laplacian
        adj_loops = adj + torch.eye(N)
        degrees = torch.sum(adj_loops, dim=0)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))
        matrix = torch.matmul(torch.matmul(D_inv_sqrt, torch.diag(degrees) - adj_loops), D_inv_sqrt)
        matrix_name = "Normalized Laplacian"
    else:
        matrix = adj
        matrix_name = "Adjacency Matrix"
    
    # Determine L
    L = args.custom_L if args.custom_L is not None else (N - args.dim)
    
    if not args.quiet:
        print(f"  Matrix: {matrix_name}")
        print(f"  L (mother wavelets): {L}")
        print(f"  dim (father wavelets): {args.dim}")
    
    # Run Baseline MMF
    if not args.quiet:
        print(f"\n  Running Baseline MMF...")
    
    model = Baseline_MMF(N, L, args.dim)
    A_rec, U, D, mother_coeff, father_coeff, mother_w, father_w = model(matrix)
    
    # Compute loss
    loss = torch.norm(matrix - A_rec, p='fro').item()
    orig_norm = torch.norm(matrix, p='fro').item()
    rec_norm = torch.norm(A_rec, p='fro').item()
    rel_error = (loss / orig_norm) * 100 if orig_norm > 0 else 0
    
    if not args.quiet or args.show_stats:
        print(f"\n  {'='*56}")
        print(f"  RESULTS")
        print(f"  {'='*56}")
        print(f"  Reconstruction Loss: {loss:.6f}")
        print(f"  Original norm:       {orig_norm:.6f}")
        print(f"  Reconstructed norm:  {rec_norm:.6f}")
        print(f"  Relative error:      {rel_error:.2f}%")
        print(f"  {'='*56}")
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Determine output prefix
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        prefix = f"{args.dataset}_mol_{molecule_id}"
    
    # Generate layout
    if not args.quiet:
        print(f"\n  Generating {args.layout} layout...")
    
    nodes_x, nodes_y = get_graph_layout(
        adj.numpy(), 
        layout=args.layout, 
        scale=args.scale,
        seed=args.seed
    )
    
    # Determine which wavelets to save
    wavelets_to_save = []
    
    if args.save_wavelets in ['mother', 'both']:
        wavelet_range = parse_range(args.wavelet_range) if args.wavelet_range else range(L)
        wavelet_range = [i for i in wavelet_range if i < L]
        
        for l in wavelet_range:
            wavelets_to_save.append(('mother', l, mother_w[l]))
    
    if args.save_wavelets in ['father', 'both']:
        wavelet_range = parse_range(args.wavelet_range) if args.wavelet_range else range(args.dim)
        wavelet_range = [i for i in wavelet_range if i < args.dim]
        
        for l in wavelet_range:
            wavelets_to_save.append(('father', l, father_w[l]))
    
    # Save wavelets
    if not args.quiet:
        print(f"  Saving {len(wavelets_to_save)} wavelets to {args.output_folder}/")
    
    for wavelet_type, idx, wavelet in wavelets_to_save:
        file_name = os.path.join(args.output_folder, f'{prefix}_{wavelet_type}_wavelet_{idx}')
        title = f'{args.dataset} Mol #{molecule_id} - {wavelet_type.capitalize()} Wavelet {idx}'
        
        draw_graph_with_wavelet(
            adj, 
            wavelet.unsqueeze(dim=0).detach().cpu().numpy(), 
            file_name,
            nodes_x=nodes_x, 
            nodes_y=nodes_y,
            title=title,
            figsize=args.figsize,
            dpi=args.dpi
        )
        plt.close()
    
    if not args.quiet:
        print(f"  ✓ Completed molecule #{molecule_id}")
    
    return {
        'molecule_id': molecule_id,
        'n_atoms': N,
        'loss': loss,
        'relative_error': rel_error,
        'n_wavelets_saved': len(wavelets_to_save)
    }


if __name__ == '__main__':
    import os
    
    # Parse arguments
    args = parse_arguments()
    
    # Print configuration
    if not args.quiet:
        print("="*60)
        print("MMF Molecular Graph Visualization")
        print("="*60)
        print(f"Dataset:        {args.dataset}")
        print(f"Data folder:    {args.data_folder}")
        print(f"Output folder:  {args.output_folder}")
        print(f"Layout:         {args.layout}")
        print(f"Matrix:         {'Normalized Laplacian' if args.use_laplacian else 'Adjacency Matrix'}")
        print("="*60)
    
    # Load dataset
    if not args.quiet:
        print(f"\nLoading {args.dataset} dataset...")
    
    data_fn = f'{args.data_folder}/{args.dataset}/{args.dataset}.dat'
    meta_fn = f'{args.data_folder}/{args.dataset}/{args.dataset}.meta'
    
    if not os.path.exists(data_fn):
        print(f"Error: Data file not found: {data_fn}")
        sys.exit(1)
    
    data = Dataset(data_fn, meta_fn)
    
    if not args.quiet:
        print(f"Loaded {data.nMolecules} molecules")
    
    # Determine which molecules to process
    if args.molecule_range:
        molecule_ids = parse_range(args.molecule_range)
        # Filter out invalid IDs
        molecule_ids = [mid for mid in molecule_ids if mid < data.nMolecules]
        if not args.quiet:
            print(f"Processing {len(molecule_ids)} molecules: {molecule_ids}")
    else:
        molecule_ids = [args.molecule_id]
    
    # Process molecules
    results = []
    for mol_id in molecule_ids:
        result = process_molecule(mol_id, data, args)
        if result:
            results.append(result)
    
    # Summary
    if not args.quiet and len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Processed {len(results)} molecules")
        print(f"Average atoms: {np.mean([r['n_atoms'] for r in results]):.1f}")
        print(f"Average reconstruction error: {np.mean([r['relative_error'] for r in results]):.2f}%")
        print(f"Total wavelets saved: {sum(r['n_wavelets_saved'] for r in results)}")
        print(f"Output folder: {args.output_folder}/")
        print(f"{'='*60}")
    
    if not args.quiet:
        print("\n✓ All done!")