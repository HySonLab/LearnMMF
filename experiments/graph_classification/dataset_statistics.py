"""
Dataset Statistics - Molecular Graph Size Analysis

This script analyzes and visualizes the size distribution and properties
of molecular graphs in graph kernel benchmark datasets.

Author: Khang Nguyen
"""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import sys

# Add source directory to path
sys.path.append('../../source/')
from Dataset import Dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze molecular graph dataset statistics'
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., MUTAG, PTC, DD, NCI1)')
    parser.add_argument('--data_folder', type=str, default='../../data/',
                        help='Data folder path')
    parser.add_argument('--output_dir', type=str, default='./statistics/',
                        help='Output directory for plots and statistics')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format for figures')
    
    return parser.parse_args()


def compute_graph_statistics(data):
    """
    Compute comprehensive statistics about molecular graphs.
    
    Args:
        data: Dataset object
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {
        'num_atoms': [],
        'num_edges': [],
        'avg_degree': [],
        'max_degree': [],
        'min_degree': [],
        'density': [],
        'num_components': [],
        'atomic_types': [],
        'labels': []
    }
    
    for mol in data.molecules:
        N = mol.nAtoms
        stats['num_atoms'].append(N)
        stats['labels'].append(mol.class_)
        
        # Count edges
        total_edges = sum(atom.nNeighbors for atom in mol.atoms)
        stats['num_edges'].append(total_edges // 2)  # Undirected graph
        
        # Degree statistics
        degrees = [atom.nNeighbors for atom in mol.atoms]
        stats['avg_degree'].append(np.mean(degrees) if degrees else 0)
        stats['max_degree'].append(max(degrees) if degrees else 0)
        stats['min_degree'].append(min(degrees) if degrees else 0)
        
        # Graph density: 2*E / (N*(N-1))
        if N > 1:
            stats['density'].append(2 * (total_edges // 2) / (N * (N - 1)))
        else:
            stats['density'].append(0)
        
        # Collect atomic types
        stats['atomic_types'].extend([atom.atomic_type for atom in mol.atoms])
    
    # Convert to numpy arrays
    for key in ['num_atoms', 'num_edges', 'avg_degree', 'max_degree', 
                'min_degree', 'density', 'labels']:
        stats[key] = np.array(stats[key])
    
    return stats


def print_statistics(data, stats):
    """Print comprehensive statistics to console."""
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nDataset: {data.data_fn.split('/')[-2]}")
    print(f"Total molecules: {data.nMolecules}")
    print(f"Number of classes: {data.nClasses}")
    print(f"Class labels: {', '.join(data.classes)}")
    
    print("\n" + "-" * 80)
    print("GRAPH SIZE STATISTICS")
    print("-" * 80)
    print(f"Total atoms across all graphs: {data.total_nAtoms}")
    print(f"Total bonds/edges: {data.total_nBonds}")
    
    print(f"\nAtoms per graph:")
    print(f"  Mean:   {np.mean(stats['num_atoms']):.2f}")
    print(f"  Median: {np.median(stats['num_atoms']):.2f}")
    print(f"  Std:    {np.std(stats['num_atoms']):.2f}")
    print(f"  Min:    {np.min(stats['num_atoms'])}")
    print(f"  Max:    {np.max(stats['num_atoms'])}")
    print(f"  Q1:     {np.percentile(stats['num_atoms'], 25):.2f}")
    print(f"  Q3:     {np.percentile(stats['num_atoms'], 75):.2f}")
    
    print(f"\nEdges per graph:")
    print(f"  Mean:   {np.mean(stats['num_edges']):.2f}")
    print(f"  Median: {np.median(stats['num_edges']):.2f}")
    print(f"  Std:    {np.std(stats['num_edges']):.2f}")
    print(f"  Min:    {np.min(stats['num_edges'])}")
    print(f"  Max:    {np.max(stats['num_edges'])}")
    
    print("\n" + "-" * 80)
    print("DEGREE STATISTICS")
    print("-" * 80)
    print(f"Average degree per graph:")
    print(f"  Mean:   {np.mean(stats['avg_degree']):.2f}")
    print(f"  Median: {np.median(stats['avg_degree']):.2f}")
    
    print(f"\nMaximum degree:")
    print(f"  Overall max: {data.max_degree}")
    print(f"  Overall min: {data.min_degree}")
    print(f"  Mean max:    {np.mean(stats['max_degree']):.2f}")
    
    print("\n" + "-" * 80)
    print("DENSITY STATISTICS")
    print("-" * 80)
    print(f"Overall dataset density: {data.density:.4f}")
    print(f"Graph density:")
    print(f"  Mean:   {np.mean(stats['density']):.4f}")
    print(f"  Median: {np.median(stats['density']):.4f}")
    print(f"  Std:    {np.std(stats['density']):.4f}")
    
    print("\n" + "-" * 80)
    print("ATOMIC TYPE STATISTICS")
    print("-" * 80)
    print(f"Number of atomic types: {data.nAtomicTypes}")
    print(f"Atomic types: {', '.join(data.all_atomic_type)}")
    
    # Count atomic type frequencies
    atomic_counter = Counter(stats['atomic_types'])
    print(f"\nAtomic type distribution:")
    for atom_type, count in atomic_counter.most_common():
        percentage = 100 * count / len(stats['atomic_types'])
        print(f"  {atom_type:>3s}: {count:6d} ({percentage:5.2f}%)")
    
    print("\n" + "-" * 80)
    print("CLASS DISTRIBUTION")
    print("-" * 80)
    label_counter = Counter(stats['labels'])
    for class_idx, count in sorted(label_counter.items()):
        percentage = 100 * count / len(stats['labels'])
        class_name = data.classes[class_idx]
        print(f"  Class {class_idx} ({class_name}): {count:4d} ({percentage:5.2f}%)")
    
    print("\n" + "-" * 80)
    print("SIZE RANGES")
    print("-" * 80)
    
    # Define size ranges with more detail for large graphs
    size_ranges = [
        (0, 10, "Tiny (1-10 atoms)"),
        (11, 20, "Small (11-20 atoms)"),
        (21, 50, "Medium (21-50 atoms)"),
        (51, 100, "Large (51-100 atoms)"),
        (101, 200, "Very large (101-200 atoms)"),
        (201, 300, "Huge (201-300 atoms)"),
        (301, 400, "Extra large (301-400 atoms)"),
        (401, 500, "Massive (401-500 atoms)"),
        (501, 750, "Giant (501-750 atoms)"),
        (751, 1000, "Enormous (751-1000 atoms)"),
        (1001, 1500, "Gigantic (1001-1500 atoms)"),
        (1501, 2000, "Colossal (1501-2000 atoms)"),
        (2001, 3000, "Titanic (2001-3000 atoms)"),
        (3001, 5000, "Monumental (3001-5000 atoms)"),
        (5001, float('inf'), "Extreme (>5000 atoms)")
    ]
    
    for min_size, max_size, label in size_ranges:
        count = np.sum((stats['num_atoms'] >= min_size) & (stats['num_atoms'] <= max_size))
        if count > 0:  # Only show ranges that have graphs
            percentage = 100 * count / len(stats['num_atoms'])
            print(f"  {label:35s}: {count:4d} ({percentage:5.2f}%)")
    
    # Additional detailed statistics for large graphs
    large_graphs = stats['num_atoms'] > 200
    if np.any(large_graphs):
        print("\n" + "-" * 80)
        print("LARGE GRAPHS (>200 atoms) - DETAILED STATISTICS")
        print("-" * 80)
        large_atoms = stats['num_atoms'][large_graphs]
        print(f"Total large graphs: {len(large_atoms)} ({100*len(large_atoms)/len(stats['num_atoms']):.2f}%)")
        print(f"Size statistics for large graphs:")
        print(f"  Mean:   {np.mean(large_atoms):.2f}")
        print(f"  Median: {np.median(large_atoms):.2f}")
        print(f"  Std:    {np.std(large_atoms):.2f}")
        print(f"  Min:    {np.min(large_atoms)}")
        print(f"  Max:    {np.max(large_atoms)}")
        print(f"  Q1:     {np.percentile(large_atoms, 25):.2f}")
        print(f"  Q3:     {np.percentile(large_atoms, 75):.2f}")
        
        # Show percentiles for large graphs
        print(f"\nPercentiles of large graphs:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(large_atoms, p)
            print(f"  {p:2d}th percentile: {val:.2f}")
    
    print("=" * 80)


def plot_size_distribution(stats, output_dir, dataset_name, dpi=300, fmt='png'):
    """Plot distribution of graph sizes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Graph Size Distribution - {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram of number of atoms
    ax = axes[0, 0]
    ax.hist(stats['num_atoms'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(stats['num_atoms']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["num_atoms"]):.1f}')
    ax.axvline(np.median(stats['num_atoms']), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(stats["num_atoms"]):.1f}')
    ax.set_xlabel('Number of Atoms', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Graph Sizes (Atoms)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot of number of atoms
    ax = axes[0, 1]
    bp = ax.boxplot(stats['num_atoms'], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax.set_ylabel('Number of Atoms', fontsize=12)
    ax.set_title('Box Plot - Graph Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Min: {np.min(stats["num_atoms"])}\n'
    stats_text += f'Q1: {np.percentile(stats["num_atoms"], 25):.1f}\n'
    stats_text += f'Median: {np.median(stats["num_atoms"]):.1f}\n'
    stats_text += f'Q3: {np.percentile(stats["num_atoms"], 75):.1f}\n'
    stats_text += f'Max: {np.max(stats["num_atoms"])}'
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Histogram of number of edges
    ax = axes[1, 0]
    ax.hist(stats['num_edges'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.mean(stats['num_edges']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["num_edges"]):.1f}')
    ax.axvline(np.median(stats['num_edges']), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(stats["num_edges"]):.1f}')
    ax.set_xlabel('Number of Edges', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Graph Sizes (Edges)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_atoms = np.sort(stats['num_atoms'])
    cumulative = np.arange(1, len(sorted_atoms) + 1) / len(sorted_atoms) * 100
    ax.plot(sorted_atoms, cumulative, linewidth=2, color='darkgreen')
    ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50th percentile')
    ax.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95th percentile')
    ax.set_xlabel('Number of Atoms', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_size_distribution.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_degree_statistics(stats, output_dir, dataset_name, dpi=300, fmt='png'):
    """Plot degree-related statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Degree Statistics - {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Average degree distribution
    ax = axes[0, 0]
    ax.hist(stats['avg_degree'], bins=40, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(np.mean(stats['avg_degree']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["avg_degree"]):.2f}')
    ax.set_xlabel('Average Degree per Graph', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Average Degrees', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Maximum degree distribution
    ax = axes[0, 1]
    ax.hist(stats['max_degree'], bins=30, edgecolor='black', alpha=0.7, color='indianred')
    ax.axvline(np.mean(stats['max_degree']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["max_degree"]):.2f}')
    ax.set_xlabel('Maximum Degree per Graph', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Maximum Degrees', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Scatter: size vs average degree
    ax = axes[1, 0]
    scatter = ax.scatter(stats['num_atoms'], stats['avg_degree'], 
                        alpha=0.5, s=20, c=stats['density'], cmap='viridis')
    ax.set_xlabel('Number of Atoms', fontsize=12)
    ax.set_ylabel('Average Degree', fontsize=12)
    ax.set_title('Graph Size vs Average Degree', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Density')
    
    # 4. Scatter: edges vs atoms
    ax = axes[1, 1]
    ax.scatter(stats['num_atoms'], stats['num_edges'], alpha=0.5, s=20, color='teal')
    # Add theoretical line for fully connected graph
    max_atoms = np.max(stats['num_atoms'])
    x_line = np.linspace(0, max_atoms, 100)
    y_line = x_line * (x_line - 1) / 2
    ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2, label='Fully connected')
    ax.set_xlabel('Number of Atoms', fontsize=12)
    ax.set_ylabel('Number of Edges', fontsize=12)
    ax.set_title('Atoms vs Edges', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_degree_statistics.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_class_distribution(stats, data, output_dir, dataset_name, dpi=300, fmt='png'):
    """Plot class distribution and size by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Class Distribution - {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Class distribution pie chart
    ax = axes[0]
    label_counter = Counter(stats['labels'])
    labels = [f"Class {i}\n({data.classes[i]})" for i in sorted(label_counter.keys())]
    sizes = [label_counter[i] for i in sorted(label_counter.keys())]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
    
    # 2. Box plot of size by class
    ax = axes[1]
    class_sizes = [stats['num_atoms'][stats['labels'] == i] 
                   for i in sorted(label_counter.keys())]
    bp = ax.boxplot(class_sizes, labels=[data.classes[i] for i in sorted(label_counter.keys())],
                    patch_artist=True)
    
    # Color each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Number of Atoms', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title('Graph Size Distribution by Class', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_class_distribution.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_atomic_types(stats, data, output_dir, dataset_name, dpi=300, fmt='png'):
    """Plot atomic type distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Count atomic types
    atomic_counter = Counter(stats['atomic_types'])
    atom_types = [item[0] for item in atomic_counter.most_common()]
    counts = [item[1] for item in atomic_counter.most_common()]
    
    # Create bar plot
    bars = ax.bar(range(len(atom_types)), counts, color='skyblue', edgecolor='black')
    
    # Color the bars with a gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(atom_types)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Atomic Type', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Atomic Type Distribution - {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(atom_types)))
    ax.set_xticklabels(atom_types, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        percentage = 100 * count / total
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_atomic_types.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_density_analysis(stats, output_dir, dataset_name, dpi=300, fmt='png'):
    """Plot density-related analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Graph Density Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Density distribution
    ax = axes[0]
    ax.hist(stats['density'], bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    ax.axvline(np.mean(stats['density']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["density"]):.3f}')
    ax.set_xlabel('Graph Density', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Graph Density', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Density vs size
    ax = axes[1]
    scatter = ax.scatter(stats['num_atoms'], stats['density'], 
                        alpha=0.5, s=20, c=stats['num_edges'], cmap='plasma')
    ax.set_xlabel('Number of Atoms', fontsize=12)
    ax.set_ylabel('Graph Density', fontsize=12)
    ax.set_title('Graph Size vs Density', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Number of Edges')
    
    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_density_analysis.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_large_graphs_analysis(stats, output_dir, dataset_name, dpi=300, fmt='png'):
    """Plot detailed analysis for large graphs (>200 atoms)."""
    large_graphs = stats['num_atoms'] > 200
    
    if not np.any(large_graphs):
        print(f"No large graphs (>200 atoms) found in {dataset_name}")
        return
    
    large_atoms = stats['num_atoms'][large_graphs]
    large_edges = stats['num_edges'][large_graphs]
    large_density = stats['density'][large_graphs]
    large_avg_degree = stats['avg_degree'][large_graphs]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Large Graphs Analysis (>200 atoms) - {dataset_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Size distribution histogram with detailed bins
    ax1 = fig.add_subplot(gs[0, :2])
    bins = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, np.max(large_atoms)+1]
    counts, edges, patches = ax1.hist(large_atoms, bins=bins, edgecolor='black', 
                                       alpha=0.7, color='steelblue')
    
    # Color bars by height
    cm = plt.cm.viridis
    norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
    for count, patch in zip(counts, patches):
        patch.set_facecolor(cm(norm(count)))
    
    ax1.axvline(np.mean(large_atoms), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(large_atoms):.1f}')
    ax1.axvline(np.median(large_atoms), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(large_atoms):.1f}')
    ax1.set_xlabel('Number of Atoms', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Size Distribution of Large Graphs', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, (count, edge) in enumerate(zip(counts, edges[:-1])):
        if count > 0:
            mid = (edges[i] + edges[i+1]) / 2
            ax1.text(mid, count, f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    # 2. Statistics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f'STATISTICS\n{"="*25}\n\n'
    stats_text += f'Count: {len(large_atoms)}\n'
    stats_text += f'Percentage: {100*len(large_atoms)/len(stats["num_atoms"]):.2f}%\n\n'
    stats_text += f'Mean: {np.mean(large_atoms):.2f}\n'
    stats_text += f'Median: {np.median(large_atoms):.2f}\n'
    stats_text += f'Std: {np.std(large_atoms):.2f}\n\n'
    stats_text += f'Min: {np.min(large_atoms)}\n'
    stats_text += f'Max: {np.max(large_atoms)}\n\n'
    stats_text += f'Q1: {np.percentile(large_atoms, 25):.2f}\n'
    stats_text += f'Q3: {np.percentile(large_atoms, 75):.2f}\n\n'
    stats_text += f'95th: {np.percentile(large_atoms, 95):.2f}\n'
    stats_text += f'99th: {np.percentile(large_atoms, 99):.2f}'
    
    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Box plot with outliers
    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot(large_atoms, vert=True, patch_artist=True, 
                     showfliers=True, notch=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax3.set_ylabel('Number of Atoms', fontsize=12)
    ax3.set_title('Box Plot with Outliers', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_atoms = np.sort(large_atoms)
    cumulative = np.arange(1, len(sorted_atoms) + 1) / len(sorted_atoms) * 100
    ax4.plot(sorted_atoms, cumulative, linewidth=2, color='darkgreen')
    
    # Add percentile markers
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(large_atoms, p)
        ax4.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax4.text(val, p, f'P{p}', fontsize=8, rotation=90, va='bottom')
    
    ax4.set_xlabel('Number of Atoms', fontsize=12)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax4.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Size vs Edges scatter
    ax5 = fig.add_subplot(gs[1, 2])
    scatter = ax5.scatter(large_atoms, large_edges, alpha=0.6, s=30, 
                         c=large_density, cmap='coolwarm')
    ax5.set_xlabel('Number of Atoms', fontsize=12)
    ax5.set_ylabel('Number of Edges', fontsize=12)
    ax5.set_title('Atoms vs Edges', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Density', fontsize=10)
    
    # 6. Density distribution
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(large_density, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax6.axvline(np.mean(large_density), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(large_density):.4f}')
    ax6.set_xlabel('Graph Density', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Density Distribution', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Average degree distribution
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.hist(large_avg_degree, bins=30, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax7.axvline(np.mean(large_avg_degree), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(large_avg_degree):.2f}')
    ax7.set_xlabel('Average Degree', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    ax7.set_title('Average Degree Distribution', fontsize=13, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Size ranges pie chart
    ax8 = fig.add_subplot(gs[2, 2])
    size_bins = [
        (200, 300, "200-300"),
        (301, 500, "301-500"),
        (501, 1000, "501-1000"),
        (1001, 2000, "1001-2000"),
        (2001, float('inf'), ">2000")
    ]
    
    range_counts = []
    range_labels = []
    for min_s, max_s, label in size_bins:
        count = np.sum((large_atoms >= min_s) & (large_atoms <= max_s))
        if count > 0:
            range_counts.append(count)
            range_labels.append(f"{label}\n({count})")
    
    if range_counts:
        colors = plt.cm.Set3(np.linspace(0, 1, len(range_counts)))
        wedges, texts, autotexts = ax8.pie(range_counts, labels=range_labels, 
                                            autopct='%1.1f%%', colors=colors, 
                                            startangle=90)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        ax8.set_title('Size Range Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_large_graphs_analysis.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()



def save_statistics_to_file(data, stats, output_dir, dataset_name):
    """Save statistics to a text file."""
    output_path = output_dir / f'{dataset_name}_statistics.txt'
    
    with open(output_path, 'w') as f:
        # Redirect print to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        
        print_statistics(data, stats)
        
        sys.stdout = old_stdout
    
    print(f"Saved: {output_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style for better-looking plots
    # plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    print("\n" + "=" * 80)
    print("MOLECULAR GRAPH DATASET ANALYSIS")
    print("=" * 80 + "\n")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data_fn = f'{args.data_folder}/{args.dataset}/{args.dataset}.dat'
    meta_fn = f'{args.data_folder}/{args.dataset}/{args.dataset}.meta'
    
    try:
        data = Dataset(data_fn, meta_fn)
        print(f"Successfully loaded {data.nMolecules} molecules\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_graph_statistics(data)
    print("Done!\n")
    
    # Print statistics
    print_statistics(data, stats)
    
    # Save statistics to file
    print("\nSaving statistics to file...")
    save_statistics_to_file(data, stats, output_dir, args.dataset)
    
    # Generate plots
    print("\nGenerating plots...")
    print("-" * 80)
    
    plot_size_distribution(stats, output_dir, args.dataset, args.dpi, args.format)
    plot_degree_statistics(stats, output_dir, args.dataset, args.dpi, args.format)
    plot_class_distribution(stats, data, output_dir, args.dataset, args.dpi, args.format)
    plot_atomic_types(stats, data, output_dir, args.dataset, args.dpi, args.format)
    plot_density_analysis(stats, output_dir, args.dataset, args.dpi, args.format)
    plot_large_graphs_analysis(stats, output_dir, args.dataset, args.dpi, args.format)
    
    print("-" * 80)
    print("\nAll plots generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()