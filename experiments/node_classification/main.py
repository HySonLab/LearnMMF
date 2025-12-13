import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
import numpy as np
import argparse
import os
import time
from tqdm import tqdm

sys.path.append('../../source/')

# Learnable MMF (use the fixed version to avoid OOM)
from learnable_mmf_model import Learnable_MMF
from heuristics import *

# +---------------------------+
# | Command-line arguments    |
# +---------------------------+

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Wavelet Network for Node Classification on Citation Graphs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer'],
                        help='Dataset name')
    parser.add_argument('--data-folder', type=str, default='../../data/',
                        help='Path to data folder')
    
    # MMF parameters
    parser.add_argument('--K', type=int, default=16,
                        help='Size of Jacobian rotation matrix')
    parser.add_argument('--L', type=int, default=None,
                        help='Number of resolutions (default: N - dim)')
    parser.add_argument('--drop', type=int, default=1,
                        help='Number of rows/columns to drop per iteration')
    parser.add_argument('--dim', type=int, default=None,
                        help='Number of father wavelets (default: auto-computed)')
    parser.add_argument('--heuristics', type=str, default='smart',
                        choices=['random', 'smart'],
                        help='Heuristics for finding wavelet indices')
    
    # MMF training parameters
    parser.add_argument('--mmf-epochs', type=int, default=10000,
                        help='Number of epochs for MMF training')
    parser.add_argument('--mmf-lr', type=float, default=1e-4,
                        help='Learning rate for MMF training')
    parser.add_argument('--mmf-early-stop', action='store_true', default=True,
                        help='Use early stopping for MMF')
    
    # WNN parameters
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of spectral convolution layers')
    parser.add_argument('--hidden-dim', type=int, default=100,
                        help='Hidden dimension for each node')
    
    # WNN training parameters
    parser.add_argument('--wnn-epochs', type=int, default=256,
                        help='Number of epochs for WNN training')
    parser.add_argument('--wnn-lr', type=float, default=1e-3,
                        help='Learning rate for WNN training')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adagrad', 'sgd'],
                        help='Optimizer for WNN')
    
    # Data split
    parser.add_argument('--split', type=str, default='MMF1',
                        choices=['MMF1', 'MMF2', 'MMF3'],
                        help='Train/val/test split: MMF1=20/20/60, MMF2=40/20/40, MMF3=60/20/20')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data split')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for models and logs')
    parser.add_argument('--name', type=str, default='experiment',
                        help='Experiment name')
    parser.add_argument('--save-mmf', action='store_true',
                        help='Save MMF model')
    parser.add_argument('--load-mmf', type=str, default=None,
                        help='Load pretrained MMF model from path')
    parser.add_argument('--save-wavelets', action='store_true',
                        help='Save mother and father wavelets to files')
    parser.add_argument('--load-wavelets', type=str, default=None,
                        help='Load wavelets from folder (skips MMF training)')
    parser.add_argument('--skip-mmf', action='store_true',
                        help='Skip MMF training (requires --load-wavelets)')
    
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


# +---------------------------+
# | Data loading              |
# +---------------------------+

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
    # Handle isolated nodes
    D = torch.where(D == 0, torch.ones_like(D), D)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
    
    # Graph Laplacian
    L = torch.diag(D) - adj
    
    # Normalized graph Laplacian
    L_norm = torch.matmul(torch.matmul(D_inv_sqrt, L), D_inv_sqrt)
    
    features = torch.Tensor(np.array(features))
    labels = torch.LongTensor(np.array(labels))
    
    print(f"Loaded {dataset}:")
    print(f"  Nodes: {N}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Classes: {len(labels_list)}")
    print(f"  Edges: {int(adj.sum().item() / 2)}")
    
    return adj, L_norm, features, labels, paper_ids, labels_list


def create_data_splits(N, labels, split_type='MMF1', seed=42):
    """
    Create train/val/test splits for node classification.
    
    Args:
        N: Number of nodes
        labels: Node labels
        split_type: 'MMF1' (20/20/60), 'MMF2' (40/20/40), or 'MMF3' (60/20/20)
        seed: Random seed
    
    Returns:
        train_mask, val_mask, test_mask: Boolean masks for splits
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define split ratios
    split_ratios = {
        'MMF1': (0.2, 0.2, 0.6),
        'MMF2': (0.4, 0.2, 0.4),
        'MMF3': (0.6, 0.2, 0.2)
    }
    
    train_ratio, val_ratio, test_ratio = split_ratios[split_type]
    
    # Create random permutation
    indices = np.random.permutation(N)
    
    # Calculate split sizes
    train_size = int(N * train_ratio)
    val_size = int(N * val_ratio)
    
    # Create masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    print(f"\nData split ({split_type}):")
    print(f"  Train: {train_mask.sum().item()} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {val_mask.sum().item()} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {test_mask.sum().item()} ({test_ratio*100:.0f}%)")
    
    return train_mask, val_mask, test_mask


# +---------------------------+
# | Wavelet Network           |
# +---------------------------+

class Wavelet_Network(nn.Module):
    """
    Wavelet Network for node classification using MMF wavelets.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, device='cpu'):
        super(Wavelet_Network, self).__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # Input layers
        self.input_layer_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Hidden layers (spectral convolution)
        self.hidden_layers = nn.ModuleList()
        for layer in range(self.num_layers):
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output layers
        self.output_layer_1 = nn.Linear(self.hidden_dim * (self.num_layers + 1), self.hidden_dim)
        self.output_layer_2 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, W, f):
        """
        Forward pass for NODE classification.
        
        Args:
            W: Wavelet basis (1, L+dim, N) - combined mother and father wavelets
            f: Node features (N, D) or (1, N, D)
        
        Returns:
            Class probabilities per node (N, C) or (1, N, C)
        """
        # Ensure batch dimension
        has_batch_dim = f.dim() == 3
        if f.dim() == 2:
            f = f.unsqueeze(0)  # (1, N, D)
        
        all_hiddens = []
        
        # Input transformation
        h = torch.tanh(self.input_layer_1(f))  # (1, N, H)
        h = torch.tanh(self.input_layer_2(h))  # (1, N, H)
        all_hiddens.append(h)
        
        # Spectral convolution layers
        for layer in range(self.num_layers):
            # Apply linear transformation
            h_transformed = self.hidden_layers[layer](h)  # (1, N, H)
            
            # Wavelet transform: W @ h  (from spatial to wavelet domain)
            h_wavelet = torch.matmul(W, h_transformed)  # (1, L+dim, H)
            
            # Inverse transform: W^T @ h_wavelet (from wavelet to spatial domain)
            h = torch.tanh(torch.matmul(W.transpose(1, 2), h_wavelet))  # (1, N, H)
            all_hiddens.append(h)
        
        # Concatenate all hidden states
        concat = torch.cat(all_hiddens, dim=2)  # (1, N, H*(num_layers+1))
        
        # Output layers
        output_1 = torch.tanh(self.output_layer_1(concat))  # (1, N, H)
        logits = self.output_layer_2(output_1)  # (1, N, C)
        
        # Remove batch dimension if input didn't have it
        if not has_batch_dim:
            logits = logits.squeeze(0)  # (N, C)
        
        return F.log_softmax(logits, dim=-1)


# +---------------------------+
# | Training functions        |
# +---------------------------+

def train_wnn(model, W, features, labels, train_mask, optimizer, device):
    """Train WNN for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(W, features)
    
    # Compute loss only on training nodes
    loss = F.nll_loss(output[train_mask], labels[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    pred = output.argmax(dim=1)
    correct = (pred[train_mask] == labels[train_mask]).sum().item()
    acc = correct / train_mask.sum().item()
    
    return loss.item(), acc


def evaluate_wnn(model, W, features, labels, mask, device):
    """Evaluate WNN."""
    model.eval()
    with torch.no_grad():
        output = model(W, features)
        loss = F.nll_loss(output[mask], labels[mask])
        pred = output.argmax(dim=1)
        correct = (pred[mask] == labels[mask]).sum().item()
        acc = correct / mask.sum().item()
    return loss.item(), acc


# +---------------------------+
# | Main function             |
# +---------------------------+

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = open(os.path.join(args.output_dir, f'{args.name}.log'), 'w')
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
    
    log_print(f"{'='*60}")
    log_print(f"Wavelet Network for Node Classification")
    log_print(f"{'='*60}")
    log_print(f"Dataset: {args.dataset}")
    log_print(f"Split: {args.split}")
    log_print(f"Device: {args.device}")
    log_print(f"{'='*60}\n")
    
    device = torch.device(args.device)
    
    # Load data
    log_print("Loading citation graph...")
    adj, L_norm, features, labels, paper_ids, labels_list = load_citation_graph(
        args.data_folder, args.dataset
    )
    
    N = L_norm.size(0)
    num_features = features.size(1)
    num_classes = len(labels_list)
    
    # Auto-compute dim and L if not provided
    if args.dim is None:
        args.dim = N - args.L  # Default: leave 700 wavelets
    if args.L is None:
        args.L = N - args.dim
    
    log_print(f"\nMMF Parameters:")
    log_print(f"  N: {N}")
    log_print(f"  K: {args.K}")
    log_print(f"  L: {args.L}")
    log_print(f"  dim: {args.dim}")
    log_print(f"  drop: {args.drop}\n")
    
    # Create data splits
    train_mask, val_mask, test_mask = create_data_splits(
        N, labels, args.split, args.seed
    )
    
    # Move data to device
    L_norm = L_norm.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    # ========================================
    # STEP 1: Get wavelet bases (train MMF or load from file)
    # ========================================
    
    # Check if we should load wavelets instead of training MMF
    if args.load_wavelets is not None or args.skip_mmf:
        if args.load_wavelets is None:
            raise ValueError('--skip-mmf requires --load-wavelets to be specified')
        
        log_print('\n' + '='*60)
        log_print('Loading Pre-computed Wavelets')
        log_print('='*60 + '\n')
        
        # Load wavelets from files
        mother_path = os.path.join(args.load_wavelets, 'mother_wavelets.pt')
        father_path = os.path.join(args.load_wavelets, 'father_wavelets.pt')
        
        if not os.path.exists(mother_path) or not os.path.exists(father_path):
            raise FileNotFoundError(
                f'Wavelet files not found in {args.load_wavelets}\n'
                f'Expected: mother_wavelets.pt and father_wavelets.pt'
            )
        
        log_print(f'Loading mother wavelets from: {mother_path}')
        log_print(f'Loading father wavelets from: {father_path}')
        
        mother_w = torch.load(mother_path, map_location=device)
        father_w = torch.load(father_path, map_location=device)
        
        log_print(f'Loaded mother wavelets: {mother_w.shape}')
        log_print(f'Loaded father wavelets: {father_w.shape}')
        
        # Compute sparsity for loaded wavelets
        mother_elements = mother_w.numel()
        mother_nonzero = (mother_w.abs() > 1e-6).sum().item()
        mother_sparsity = 100 * (1 - mother_nonzero / mother_elements)
        log_print(f'Mother wavelets: {mother_w.shape[0]} wavelets, {100 - mother_sparsity:.2f}% non-zero')
        
        father_elements = father_w.numel()
        father_nonzero = (father_w.abs() > 1e-6).sum().item()
        father_sparsity = 100 * (1 - father_nonzero / father_elements)
        log_print(f'Father wavelets: {father_w.shape[0]} wavelets, {100 - father_sparsity:.2f}% non-zero')
        
        total_elements = mother_elements + father_elements
        total_nonzero = mother_nonzero + father_nonzero
        combined_sparsity = 100 * (1 - total_nonzero / total_elements)
        log_print(f'Combined sparsity: {100 - combined_sparsity:.2f}% non-zero')
        
    else:
        # Train MMF to compute wavelets
        log_print('\n' + '='*60)
        log_print('STEP 1: Learning MMF Wavelet Bases')
        log_print('='*60 + '\n')
        
        # Compute heuristics
        log_print(f"Computing {args.heuristics} heuristics...")
        A_sparse = L_norm.to_sparse()
        
        if args.heuristics == 'random':
            wavelet_indices, rest_indices = heuristics_random(
                A_sparse, args.L, args.K, args.drop, args.dim
            )
        else:
            if args.drop == 1:
                wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(
                    A_sparse, args.L, args.K, args.drop, args.dim
                )
            else:
                wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(
                    A_sparse, args.L, args.K, args.drop, args.dim
                )
        
        # Create MMF model
        log_print("Creating Learnable MMF model...")
        mmf_model = Learnable_MMF(
            L_norm, args.L, args.K, args.drop, args.dim,
            wavelet_indices, rest_indices,
            device=device
        )
        mmf_model = mmf_model.to(device)
        
        # Train MMF
        log_print(f"\nTraining MMF for {args.mmf_epochs} epochs...")
        mmf_optimizer = Adagrad(mmf_model.parameters(), lr=args.mmf_lr)
        
        all_losses = []
        all_errors = []
        norm = torch.norm(L_norm, p='fro')
        best = 1e9
        
        for epoch in range(args.mmf_epochs):
            t = time.time()
            mmf_optimizer.zero_grad()
            
            A_rec, U, D, mother_coeff, father_coeff, mother_w, father_w = mmf_model()
            
            loss = torch.norm(L_norm - A_rec, p='fro')
            loss.backward()
            
            error = loss / norm
            all_losses.append(loss.item())
            all_errors.append(error.item())
            
            if epoch % 100 == 0 or epoch == args.mmf_epochs - 1:
                log_print(f'Epoch {epoch:4d} | Loss: {loss.item():.6f} | '
                         f'Error: {error.item():.6f} | Time: {time.time()-t:.2f}s')
            
            # Early stopping
            if args.mmf_early_stop:
                if loss.item() < best:
                    best = loss.item()
                else:
                    log_print(f"Early stopping at epoch {epoch}")
                    break
            
            # Manual Cayley update
            for l in range(args.L):
                X = mmf_model.all_O[l].data.clone()
                G = mmf_model.all_O[l].grad.data.clone()
                Z = torch.matmul(G, X.transpose(0, 1)) - torch.matmul(X, G.transpose(0, 1))
                tau = args.mmf_lr
                Y = torch.matmul(
                    torch.matmul(
                        torch.inverse(torch.eye(args.K, device=device) + tau / 2 * Z),
                        torch.eye(args.K, device=device) - tau / 2 * Z
                    ),
                    X
                )
                mmf_model.all_O[l].data = Y.data
        
        # Get final wavelets
        log_print("\nExtracting learned wavelets...")
        with torch.no_grad():
            A_rec, U, D, mother_coeff, father_coeff, mother_w, father_w = mmf_model()
        
        # Compute sparsity for both mother and father wavelets
        mother_elements = mother_w.numel()
        mother_nonzero = (mother_w.abs() > 1e-6).sum().item()
        mother_sparsity = 100 * (1 - mother_nonzero / mother_elements)
        log_print(f"Mother wavelets: {mother_w.shape[0]} wavelets, {100 - mother_sparsity:.2f}% non-zero")
        
        father_elements = father_w.numel()
        father_nonzero = (father_w.abs() > 1e-6).sum().item()
        father_sparsity = 100 * (1 - father_nonzero / father_elements)
        log_print(f"Father wavelets: {father_w.shape[0]} wavelets, {100 - father_sparsity:.2f}% non-zero")
        
        # Combined sparsity
        total_elements = mother_elements + father_elements
        total_nonzero = mother_nonzero + father_nonzero
        combined_sparsity = 100 * (1 - total_nonzero / total_elements)
        log_print(f"Combined sparsity: {100 - combined_sparsity:.2f}% non-zero")
        
        # Save MMF if requested
        if args.save_mmf:
            mmf_path = os.path.join(args.output_dir, f'{args.name}_mmf.pt')
            torch.save(mmf_model.state_dict(), mmf_path)
            log_print(f"Saved MMF model to {mmf_path}")
        
        # Save wavelets if requested
        if args.save_wavelets:
            wavelets_dir = os.path.join(args.output_dir, 'wavelets')
            os.makedirs(wavelets_dir, exist_ok=True)
            
            mother_path = os.path.join(wavelets_dir, 'mother_wavelets.pt')
            father_path = os.path.join(wavelets_dir, 'father_wavelets.pt')
            
            torch.save(mother_w, mother_path)
            torch.save(father_w, father_path)
            
            log_print(f"\nSaved wavelets to {wavelets_dir}/")
            log_print(f"  - mother_wavelets.pt: {mother_w.shape}")
            log_print(f"  - father_wavelets.pt: {father_w.shape}")
        
        # ========================================
    # STEP 2: Train Wavelet Network
    # ========================================
    
    log_print("STEP 2: Training Wavelet Network")
    log_print("="*60 + "\n")
    
    # Prepare wavelet basis (concatenate mother and father wavelets)
    # Following train_wavelet_network.py: bases = torch.cat([mother, father], dim=0)
    # Mother wavelets: (L, N)
    # Father wavelets: (dim, N)
    # Combined bases: (L + dim, N)
    log_print(f"Combining wavelets: {mother_w.shape[0]} mother + {father_w.shape[0]} father = {mother_w.shape[0] + father_w.shape[0]} total")
    
    # Concatenate mother and father wavelets
    bases = torch.cat([mother_w, father_w], dim=0)  # Shape: (L+dim, N)
    
    # Add batch dimension: (1, L+dim, N)
    W = bases.unsqueeze(0).to(device)
    
    log_print(f"Wavelet basis shape: {W.shape}")
    
    # Create WNN
    log_print("Creating Wavelet Network...")
    wnn_model = Wavelet_Network(
        num_layers=args.num_layers,
        input_dim=num_features,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        device=device
    ).to(device)
    
    log_print(f"  Layers: {args.num_layers}")
    log_print(f"  Input dim: {num_features}")
    log_print(f"  Hidden dim: {args.hidden_dim}")
    log_print(f"  Output dim: {num_classes}\n")
    
    # Create optimizer
    if args.optimizer == 'adam':
        wnn_optimizer = Adam(wnn_model.parameters(), lr=args.wnn_lr)
    elif args.optimizer == 'adagrad':
        wnn_optimizer = Adagrad(wnn_model.parameters(), lr=args.wnn_lr)
    else:
        wnn_optimizer = torch.optim.SGD(wnn_model.parameters(), lr=args.wnn_lr)
    
    # Training loop
    log_print(f"Training WNN for {args.wnn_epochs} epochs...")
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(args.wnn_epochs):
        # Train
        train_loss, train_acc = train_wnn(
            wnn_model, W, features, labels, train_mask, wnn_optimizer, device
        )
        
        # Validate
        val_loss, val_acc = evaluate_wnn(
            wnn_model, W, features, labels, val_mask, device
        )
        
        # Test
        test_loss, test_acc = evaluate_wnn(
            wnn_model, W, features, labels, test_mask, device
        )
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            # Save best model
            torch.save(
                wnn_model.state_dict(),
                os.path.join(args.output_dir, f'{args.name}_best.pt')
            )
        
        # Log progress
        if epoch % 10 == 0 or epoch == args.wnn_epochs - 1:
            log_print(
                f'Epoch {epoch:3d} | '
                f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
                f'Test Acc: {test_acc:.4f} | '
                f'Best Val: {best_val_acc:.4f} Test: {best_test_acc:.4f}'
            )
    
    # Final results
    log_print("FINAL RESULTS")
    log_print("="*60)
    log_print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    log_print(f"Test Accuracy (at best val): {best_test_acc:.4f}")
    log_print("="*60 + "\n")
    
    log_file.close()
    
    return best_test_acc


if __name__ == '__main__':
    main()