import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx
import argparse
import os
from tqdm import tqdm

sys.path.append('../../source/')

from baseline_mmf_model import Baseline_MMF
from heuristics import *

# Set style for better-looking plots
sns.set_palette("husl")


def load_citation_graph(data_folder, dataset):
    """
    Load citation graph dataset (Cora or Citeseer).
    
    Returns:
        adj: Adjacency matrix (N x N)
        L_norm: Normalized graph Laplacian (N x N)
        features: Node features (N x D)
        labels: Node labels (N,)
        paper_ids: List of paper IDs
        labels_list: List of unique label names
    """
    cites_fn = os.path.join(data_folder, dataset, f'{dataset}.cites')
    content_fn = os.path.join(data_folder, dataset, f'{dataset}.content')
    
    # Read content file
    paper_ids = []
    labels = []
    labels_list = []
    features = []
    
    with open(content_fn) as file:
        for line in file:
            words = line.split()
            paper_ids.append(words[0])
            label = words[-1]
            if label not in labels_list:
                labels_list.append(label)
            labels.append(labels_list.index(label))
            feature = [float(w) for w in words[1:-1]]
            features.append(feature)
    
    # Read cites file
    N = len(paper_ids)
    adj = np.zeros((N, N))
    
    with open(cites_fn) as file:
        for line in file:
            words = line.split()
            if len(words) != 2:
                continue
            if words[0] in paper_ids and words[1] in paper_ids:
                id1 = paper_ids.index(words[0])
                id2 = paper_ids.index(words[1])
                adj[id1, id2] = 1
                adj[id2, id1] = 1
    
    # Convert to tensors
    adj = torch.Tensor(adj)
    
    # Compute normalized Laplacian
    D = torch.sum(adj, dim=0)
    D = torch.where(D == 0, torch.ones_like(D), D)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
    
    L = torch.diag(D) - adj
    L_norm = torch.matmul(torch.matmul(D_inv_sqrt, L), D_inv_sqrt)
    
    features = torch.Tensor(np.array(features))
    labels = torch.LongTensor(np.array(labels))
    
    print(f"\nLoaded {dataset}:")
    print(f"  Nodes: {N}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Classes: {len(labels_list)}")
    print(f"  Edges: {int(adj.sum().item() / 2)}")
    
    return adj, L_norm, features, labels, paper_ids, labels_list


def compute_localization_metrics(wavelet, adj):
    """
    Compute localization metrics for a single wavelet.
    
    Args:
        wavelet: (N,) tensor representing the wavelet coefficients
        adj: (N, N) adjacency matrix
    
    Returns:
        Dictionary with localization metrics
    """
    N = wavelet.shape[0]
    wavelet_np = wavelet.cpu().numpy()
    adj_np = adj.cpu().numpy()
    
    # Find the center node (maximum absolute coefficient)
    center_node = np.argmax(np.abs(wavelet_np))
    center_value = np.abs(wavelet_np[center_node])
    
    # Compute energy concentration
    sorted_coeffs = np.sort(np.abs(wavelet_np))[::-1]
    cumsum = np.cumsum(sorted_coeffs)
    total_energy = cumsum[-1]
    
    # Find number of nodes containing 90% energy
    idx_90 = np.where(cumsum >= 0.9 * total_energy)[0]
    support_90 = idx_90[0] + 1 if len(idx_90) > 0 else N
    
    # Find number of nodes containing 95% energy
    idx_95 = np.where(cumsum >= 0.95 * total_energy)[0]
    support_95 = idx_95[0] + 1 if len(idx_95) > 0 else N
    
    # Compute graph distances from center
    if np.sum(adj_np) > 0:
        adj_sparse = csr_matrix(adj_np)
        distances = shortest_path(adj_sparse, indices=center_node, directed=False)
        
        # Average distance weighted by coefficient magnitude
        weights = np.abs(wavelet_np)
        weighted_dist = np.sum(distances * weights) / np.sum(weights)
        
        # Maximum distance with significant coefficients (> 1% of max)
        significant = np.abs(wavelet_np) > 0.01 * center_value
        max_dist = np.max(distances[significant]) if np.any(significant) else 0
    else:
        weighted_dist = 0
        max_dist = 0
    
    return {
        'center_node': center_node,
        'center_value': center_value,
        'support_90': support_90,
        'support_95': support_95,
        'support_ratio_90': support_90 / N,
        'support_ratio_95': support_95 / N,
        'weighted_distance': weighted_dist,
        'max_distance': max_dist,
        'sparsity': np.sum(np.abs(wavelet_np) > 1e-6) / N
    }


def visualize_wavelet_hierarchy(mother_w, father_w, adj, labels, save_dir, num_examples=8):
    """
    Visualize the hierarchical structure of wavelets across different scales.
    """
    print("\n" + "="*60)
    print("VISUALIZING HIERARCHICAL STRUCTURE")
    print("="*60)
    
    L = mother_w.shape[0]
    N = mother_w.shape[1]
    
    # Select wavelets to visualize at different scales
    scales_to_show = np.linspace(0, L-1, min(num_examples, L), dtype=int)
    
    fig, axes = plt.subplots(2, len(scales_to_show)//2, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, scale in enumerate(scales_to_show):
        ax = axes[idx]
        
        wavelet = mother_w[scale].cpu().numpy()
        
        # Compute localization metrics
        metrics = compute_localization_metrics(mother_w[scale], adj)
        center = metrics['center_node']
        
        # Sort nodes by distance from center for better visualization
        adj_np = adj.cpu().numpy()
        if np.sum(adj_np) > 0:
            adj_sparse = csr_matrix(adj_np)
            distances = shortest_path(adj_sparse, indices=center, directed=False)
            sort_idx = np.argsort(distances)
            wavelet_sorted = wavelet[sort_idx]
        else:
            wavelet_sorted = wavelet
        
        # Plot
        im = ax.imshow(wavelet_sorted.reshape(1, -1), aspect='auto', cmap='RdBu_r', 
                      vmin=-np.abs(wavelet).max(), vmax=np.abs(wavelet).max())
        ax.set_title(f'Scale {scale+1}/{L}\nSupport: {metrics["support_90"]}/{N} nodes ({metrics["support_ratio_90"]*100:.1f}%)', 
                    fontsize=10)
        ax.set_xlabel('Nodes (sorted by distance)', fontsize=8)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hierarchy_scales.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: hierarchy_scales.png")
    plt.close()


def visualize_localization_on_graph(mother_w, adj, labels, save_dir, num_examples=6):
    """
    Visualize wavelets overlaid on the actual graph structure.
    """
    print("\n" + "="*60)
    print("VISUALIZING LOCALIZATION ON GRAPH")
    print("="*60)
    
    L = mother_w.shape[0]
    N = mother_w.shape[1]
    
    # Convert to NetworkX graph for visualization
    adj_np = adj.cpu().numpy()
    G = nx.from_numpy_array(adj_np)
    
    # Compute layout (this might take a while for large graphs)
    print("Computing graph layout...")
    pos = nx.spring_layout(G, k=1/np.sqrt(N), iterations=50, seed=42)
    
    # Select wavelets at different scales
    scales_to_show = np.linspace(0, L-1, min(num_examples, L), dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, scale in enumerate(scales_to_show[:6]):
        ax = axes[idx]
        
        wavelet = mother_w[scale].cpu().numpy()
        metrics = compute_localization_metrics(mother_w[scale], adj)
        
        # Node colors based on wavelet coefficients
        node_colors = wavelet
        node_sizes = np.abs(wavelet) * 500 + 10
        
        # Draw graph
        nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax)
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                       node_size=node_sizes, cmap='RdBu_r',
                                       vmin=-np.abs(wavelet).max(), 
                                       vmax=np.abs(wavelet).max(), ax=ax)
        
        # Highlight center
        center = metrics['center_node']
        nx.draw_networkx_nodes(G, pos, nodelist=[center], 
                             node_color='yellow', node_size=200, 
                             node_shape='*', ax=ax)
        
        ax.set_title(f'Scale {scale+1}/{L} - Localized around node {center}\n'
                    f'Support: {metrics["support_90"]} nodes ({metrics["support_ratio_90"]*100:.1f}%)',
                    fontsize=10)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'localization_on_graph.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: localization_on_graph.png")
    plt.close()


def visualize_support_statistics(mother_w, father_w, adj, save_dir):
    """
    Visualize statistics about wavelet support and localization.
    """
    print("\n" + "="*60)
    print("COMPUTING LOCALIZATION STATISTICS")
    print("="*60)
    
    L = mother_w.shape[0]
    dim = father_w.shape[0]
    N = mother_w.shape[1]
    
    # Compute metrics for all mother wavelets
    mother_metrics = []
    print("Analyzing mother wavelets...")
    for i in tqdm(range(L)):
        metrics = compute_localization_metrics(mother_w[i], adj)
        metrics['scale'] = i
        metrics['type'] = 'mother'
        mother_metrics.append(metrics)
    
    # Compute metrics for all father wavelets
    father_metrics = []
    print("Analyzing father wavelets...")
    for i in tqdm(range(dim)):
        metrics = compute_localization_metrics(father_w[i], adj)
        metrics['scale'] = i
        metrics['type'] = 'father'
        father_metrics.append(metrics)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Support size across scales
    ax = axes[0, 0]
    scales = [m['scale'] for m in mother_metrics]
    support_90 = [m['support_90'] for m in mother_metrics]
    support_95 = [m['support_95'] for m in mother_metrics]
    
    ax.plot(scales, support_90, 'o-', label='90% energy', linewidth=2, markersize=6)
    ax.plot(scales, support_95, 's-', label='95% energy', linewidth=2, markersize=6)
    ax.axhline(N, color='r', linestyle='--', label=f'Total nodes ({N})', alpha=0.5)
    ax.set_xlabel('Scale (coarse → fine)', fontsize=11)
    ax.set_ylabel('Number of Nodes', fontsize=11)
    ax.set_title('Support Size Across Scales', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Support ratio across scales
    ax = axes[0, 1]
    support_ratio_90 = [m['support_ratio_90'] * 100 for m in mother_metrics]
    support_ratio_95 = [m['support_ratio_95'] * 100 for m in mother_metrics]
    
    ax.plot(scales, support_ratio_90, 'o-', label='90% energy', linewidth=2, markersize=6)
    ax.plot(scales, support_ratio_95, 's-', label='95% energy', linewidth=2, markersize=6)
    ax.set_xlabel('Scale (coarse → fine)', fontsize=11)
    ax.set_ylabel('Percentage of Nodes (%)', fontsize=11)
    ax.set_title('Localization: Support as % of Total Nodes', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Sparsity across scales
    ax = axes[0, 2]
    sparsity = [m['sparsity'] * 100 for m in mother_metrics]
    ax.plot(scales, sparsity, 'o-', color='green', linewidth=2, markersize=6)
    ax.set_xlabel('Scale (coarse → fine)', fontsize=11)
    ax.set_ylabel('Non-zero Coefficients (%)', fontsize=11)
    ax.set_title('Sparsity Across Scales', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Histogram of support sizes
    ax = axes[1, 0]
    all_support = support_90 + [m['support_90'] for m in father_metrics]
    ax.hist(all_support, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(all_support), color='r', linestyle='--', 
              label=f'Mean: {np.mean(all_support):.1f}', linewidth=2)
    ax.set_xlabel('Support Size (90% energy)', fontsize=11)
    ax.set_ylabel('Number of Wavelets', fontsize=11)
    ax.set_title('Distribution of Wavelet Support Sizes', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Weighted distance from center
    ax = axes[1, 1]
    weighted_dist = [m['weighted_distance'] for m in mother_metrics]
    ax.plot(scales, weighted_dist, 'o-', color='purple', linewidth=2, markersize=6)
    ax.set_xlabel('Scale (coarse → fine)', fontsize=11)
    ax.set_ylabel('Average Graph Distance', fontsize=11)
    ax.set_title('Spatial Spread: Weighted Distance from Center', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Mother vs Father comparison
    ax = axes[1, 2]
    mother_support_avg = np.mean([m['support_ratio_90'] for m in mother_metrics]) * 100
    father_support_avg = np.mean([m['support_ratio_90'] for m in father_metrics]) * 100
    
    bars = ax.bar(['Mother\nWavelets', 'Father\nWavelets'], 
                  [mother_support_avg, father_support_avg],
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Average Support (% of nodes)', fontsize=11)
    ax.set_title('Mother vs Father Wavelet Localization', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'localization_statistics.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: localization_statistics.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LOCALIZATION SUMMARY")
    print("="*60)
    print(f"\nMother Wavelets (L={L}):")
    print(f"  Average support (90% energy): {mother_support_avg:.2f}% of nodes")
    print(f"  Average support (95% energy): {np.mean([m['support_ratio_95'] for m in mother_metrics])*100:.2f}% of nodes")
    print(f"  Average weighted distance: {np.mean([m['weighted_distance'] for m in mother_metrics]):.2f}")
    
    print(f"\nFather Wavelets (dim={dim}):")
    print(f"  Average support (90% energy): {father_support_avg:.2f}% of nodes")
    print(f"  Average support (95% energy): {np.mean([m['support_ratio_95'] for m in father_metrics])*100:.2f}% of nodes")
    print(f"  Average weighted distance: {np.mean([m['weighted_distance'] for m in father_metrics]):.2f}")
    
    print("\nInterpretation:")
    print("  - Lower support % = better localization")
    print("  - Mother wavelets capture multi-scale structure")
    print("  - Father wavelets provide low-frequency basis")


def visualize_wavelet_matrix(mother_w, father_w, save_dir):
    """
    Visualize the complete wavelet matrix showing hierarchical structure.
    """
    print("\n" + "="*60)
    print("VISUALIZING COMPLETE WAVELET BASIS MATRIX")
    print("="*60)
    
    L = mother_w.shape[0]
    dim = father_w.shape[0]
    N = mother_w.shape[1]
    
    # Combine mother and father wavelets
    W = torch.cat([mother_w, father_w], dim=0)
    W_np = W.cpu().numpy()
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Full wavelet matrix
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(W_np, aspect='auto', cmap='RdBu_r', 
                    vmin=-np.abs(W_np).max(), vmax=np.abs(W_np).max())
    ax1.axhline(L-0.5, color='yellow', linewidth=2, linestyle='--', 
               label='Mother/Father boundary')
    ax1.set_xlabel('Nodes', fontsize=11)
    ax1.set_ylabel('Wavelet Index', fontsize=11)
    ax1.set_title(f'Complete Wavelet Basis Matrix\n({L} mother + {dim} father = {L+dim} total wavelets)', 
                 fontsize=12, fontweight='bold')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # 2. Mother wavelets only (hierarchical structure)
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(mother_w.cpu().numpy(), aspect='auto', cmap='RdBu_r',
                    vmin=-np.abs(mother_w.cpu().numpy()).max(), 
                    vmax=np.abs(mother_w.cpu().numpy()).max())
    ax2.set_xlabel('Nodes', fontsize=11)
    ax2.set_ylabel('Scale (coarse → fine)', fontsize=11)
    ax2.set_title(f'Mother Wavelets: Hierarchical Multi-Scale Basis\n(L={L} scales)', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Father wavelets only
    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(father_w.cpu().numpy(), aspect='auto', cmap='RdBu_r',
                    vmin=-np.abs(father_w.cpu().numpy()).max(), 
                    vmax=np.abs(father_w.cpu().numpy()).max())
    ax3.set_xlabel('Nodes', fontsize=11)
    ax3.set_ylabel('Father Wavelet Index', fontsize=11)
    ax3.set_title(f'Father Wavelets: Low-Frequency Basis\n(dim={dim} wavelets)', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Sparsity pattern (binary: significant vs insignificant coefficients)
    ax4 = plt.subplot(2, 2, 4)
    threshold = 0.01 * np.abs(W_np).max()
    sparsity_pattern = (np.abs(W_np) > threshold).astype(float)
    im4 = ax4.imshow(sparsity_pattern, aspect='auto', cmap='binary', vmin=0, vmax=1)
    ax4.axhline(L-0.5, color='red', linewidth=2, linestyle='--')
    ax4.set_xlabel('Nodes', fontsize=11)
    ax4.set_ylabel('Wavelet Index', fontsize=11)
    total_elements = W_np.size
    nonzero_elements = np.sum(sparsity_pattern)
    sparsity_percent = (1 - nonzero_elements / total_elements) * 100
    ax4.set_title(f'Sparsity Pattern (threshold={threshold:.4f})\n'
                 f'Sparsity: {sparsity_percent:.1f}% zero coefficients', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4, ticks=[0, 1], label='0=zero, 1=significant')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'wavelet_basis_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: wavelet_basis_matrix.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize MMF Basis Localization and Hierarchy')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='cora',
                       choices=['cora', 'citeseer'],
                       help='Dataset name')
    parser.add_argument('--data-folder', type=str, default='../../data/',
                       help='Path to data folder')
    
    # MMF parameters
    parser.add_argument('--K', type=int, default=2,
                       help='Size of Jacobian rotation matrix')
    parser.add_argument('--L', type=int, default=None,
                       help='Number of resolutions')
    parser.add_argument('--drop', type=int, default=1,
                       help='Number of rows/columns to drop per iteration')
    parser.add_argument('--dim', type=int, default=2,
                       help='Number of father wavelets')
    parser.add_argument('--heuristics', type=str, default='smart',
                       choices=['random', 'smart'],
                       help='Heuristics for finding wavelet indices')
    
    # MMF training parameters
    parser.add_argument('--mmf-epochs', type=int, default=5000,
                       help='Number of epochs for MMF training')
    parser.add_argument('--mmf-lr', type=float, default=1e-4,
                       help='Learning rate for MMF training')
    parser.add_argument('--mmf-early-stop', action='store_true', default=True,
                       help='Use early stopping for MMF')
    
    # Loading parameters
    parser.add_argument('--load-wavelets', type=str, default=None,
                       help='Load wavelets from folder (skips MMF training)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load graph
    print("\n" + "="*60)
    print("LOADING GRAPH DATA")
    print("="*60)
    adj, L_norm, features, labels, paper_ids, labels_list = load_citation_graph(
        args.data_folder, args.dataset
    )
    
    N = L_norm.shape[0]
    L_norm = L_norm.to(device)
    adj = adj.to(device)
    
    # Set default parameters
    if args.L is None:
        args.L = N - (args.dim if args.dim is not None else 0)
    if args.dim is None:
        args.dim = N - args.L
    
    print(f"\nMMF Parameters:")
    print(f"  L (number of scales): {args.L}")
    print(f"  dim (father wavelets): {args.dim}")
    print(f"  K (rotation size): {args.K}")
    print(f"  drop: {args.drop}")
    
    # Load or compute wavelets
    if args.load_wavelets:
        print("\n" + "="*60)
        print("LOADING PRECOMPUTED WAVELETS")
        print("="*60)
        
        mother_path = os.path.join(args.load_wavelets, 'mother_wavelets.pt')
        father_path = os.path.join(args.load_wavelets, 'father_wavelets.pt')
        
        mother_w = torch.load(mother_path).to(device)
        father_w = torch.load(father_path).to(device)
        
        print(f"Loaded mother wavelets: {mother_w.shape}")
        print(f"Loaded father wavelets: {father_w.shape}")
    else:
        print("\n" + "="*60)
        print("COMPUTING MMF BASIS")
        print("="*60)
        
        # Compute wavelet indices using heuristics
        A_sparse = L_norm.to_sparse()
        
        print(f"\nComputing wavelet indices using '{args.heuristics}' heuristics...")
        if args.heuristics == 'random':
            wavelet_indices, rest_indices = heuristics_random_multiple_wavelets(
                A_sparse, args.L, args.K, args.drop, args.dim
            )
        else:
            wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(
                A_sparse, args.L, args.K, args.drop, args.dim
            )
        
        # Create and train MMF model
        print("\nCreating Learnable MMF model...")
        # mmf_model = Learnable_MMF(
        #     L_norm, args.L, args.K, args.drop, args.dim,
        #     wavelet_indices, rest_indices,
        #     device=device
        # )
        # mmf_model = mmf_model.to(device)
        
        # # Train MMF
        # print(f"\nTraining MMF for {args.mmf_epochs} epochs...")
        # from torch.optim import Adagrad
        # mmf_optimizer = Adagrad(mmf_model.parameters(), lr=args.mmf_lr)
        
        # norm = torch.norm(L_norm, p='fro')
        # best = 1e9
        
        # for epoch in tqdm(range(args.mmf_epochs), desc="Training MMF"):
        #     mmf_optimizer.zero_grad()
            
        #     A_rec, U, D, mother_coeff, father_coeff, mother_w, father_w = mmf_model()
            
        #     loss = torch.norm(L_norm - A_rec, p='fro')
        #     loss.backward()
            
        #     error = loss / norm
            
        #     if epoch % 500 == 0:
        #         print(f'\nEpoch {epoch:4d} | Loss: {loss.item():.6f} | Error: {error.item():.6f}')
            
        #     # Early stopping
        #     if args.mmf_early_stop:
        #         if loss.item() < best:
        #             best = loss.item()
        #         else:
        #             print(f"\nEarly stopping at epoch {epoch}")
        #             break
            
        #     # Manual Cayley update
        #     for l in range(args.L):
        #         X = mmf_model.all_O[l].data.clone()
        #         G = mmf_model.all_O[l].grad.data.clone()
        #         Z = torch.matmul(G, X.transpose(0, 1)) - torch.matmul(X, G.transpose(0, 1))
        #         tau = args.mmf_lr
        #         Y = torch.matmul(
        #             torch.matmul(
        #                 torch.inverse(torch.eye(args.K, device=device) + tau / 2 * Z),
        #                 torch.eye(args.K, device=device) - tau / 2 * Z
        #             ),
        #             X
        #         )
        #         mmf_model.all_O[l].data = Y.data
        
        # # Get final wavelets
        # print("\nExtracting learned wavelets...")
        # with torch.no_grad():
        #     A_rec, U, D, mother_coeff, father_coeff, mother_w, father_w = mmf_model()

        mmf_model =  Baseline_MMF(
            L_norm.size(0), args.L, args.dim, device=device
        )
        A_rec, U, D, mother_coeff, father_coeff, mother_w, father_w = mmf_model(L_norm)
    
    # Generate all visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualize_wavelet_matrix(mother_w, father_w, args.output_dir)
    visualize_wavelet_hierarchy(mother_w, father_w, adj, labels, args.output_dir)
    visualize_support_statistics(mother_w, father_w, adj, args.output_dir)
    
    # Only visualize on graph for smaller datasets
    if N <= 1000:
        visualize_localization_on_graph(mother_w, adj, labels, args.output_dir)
    else:
        print(f"\nSkipping graph visualization (N={N} too large for clear visualization)")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nAll visualizations saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  1. wavelet_basis_matrix.png - Complete basis showing hierarchy")
    print("  2. hierarchy_scales.png - Wavelets at different scales")
    print("  3. localization_statistics.png - Quantitative analysis")
    if N <= 1000:
        print("  4. localization_on_graph.png - Wavelets on graph structure")


if __name__ == '__main__':
    main()