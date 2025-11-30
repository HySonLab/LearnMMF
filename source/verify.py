"""
Verification Script: CPU vs CUDA Learnable MMF

This script verifies that the CUDA version produces identical results to the CPU version.

Usage:
    python verify_cuda_learnable_mmf.py
"""

import torch
import numpy as np
import time

# Import CPU version (original)
from learnable_mmf_model import learnable_mmf_train

# Import CUDA version
from learnable_mmf_model_cuda import learnable_mmf_train_cuda


def generate_test_matrix(N=50, seed=42):
    """Generate a symmetric test matrix."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    A = torch.randn(N, N)
    A = A + A.t()  # Make symmetric
    
    return A


def generate_test_indices(N, L, K, seed=42):
    """Generate random wavelet and rest indices for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Wavelet indices
    wavelet_indices = torch.randperm(N)[:L].tolist()
    wavelet_indices = [[idx] for idx in wavelet_indices]
    
    # Rest indices (K-1 neighbors for each wavelet)
    rest_indices = []
    for wavelet_idx in wavelet_indices:
        # Create mask excluding wavelet index
        mask = torch.ones(N, dtype=torch.bool)
        mask[wavelet_idx[0]] = False
        
        # Sample K-1 rest indices
        valid_indices = torch.arange(N)[mask]
        rest = valid_indices[torch.randperm(N-1)[:K-1]].tolist()
        rest_indices.append(rest)
    
    return wavelet_indices, rest_indices


def compare_tensors(tensor_cpu, tensor_cuda, name, rtol=1e-5, atol=1e-8):
    """Compare CPU and CUDA tensors."""
    # Move CUDA tensor to CPU for comparison
    tensor_cuda_cpu = tensor_cuda.cpu()
    
    # Check shapes
    if tensor_cpu.shape != tensor_cuda_cpu.shape:
        print(f"✗ {name}: Shape mismatch!")
        print(f"  CPU shape: {tensor_cpu.shape}")
        print(f"  CUDA shape: {tensor_cuda_cpu.shape}")
        return False
    
    # Check values
    max_diff = torch.max(torch.abs(tensor_cpu - tensor_cuda_cpu)).item()
    rel_diff = max_diff / (torch.max(torch.abs(tensor_cpu)).item() + 1e-10)
    
    is_close = torch.allclose(tensor_cpu, tensor_cuda_cpu, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✓ {name}: MATCH (max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e})")
        return True
    else:
        print(f"✗ {name}: MISMATCH (max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e})")
        return False


def run_verification(N=50, L=40, K=2, epochs=100, learning_rate=1e-4):
    """Run verification comparing CPU and CUDA versions."""
    print("=" * 80)
    print("LEARNABLE MMF: CPU vs CUDA VERIFICATION")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("This verification requires a CUDA-capable GPU.")
        return False
    
    print(f"\nCUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Configuration: N={N}, L={L}, K={K}, epochs={epochs}")
    print("-" * 80)
    
    # Generate test data
    print("\nGenerating test data...")
    A = generate_test_matrix(N)
    wavelet_indices, rest_indices = generate_test_indices(N, L, K)
    
    drop = 1
    dim = N - L * drop
    
    print(f"  Matrix size: {N}x{N}")
    print(f"  Wavelet levels: {L}")
    print(f"  Rotation size: {K}")
    print(f"  Final dimension: {dim}")
    
    # Run CPU version
    print("\n" + "-" * 80)
    print("Running CPU version...")
    print("-" * 80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    start_cpu = time.time()
    A_rec_cpu, U_cpu, D_cpu, mother_coeffs_cpu, father_coeffs_cpu, \
        mother_wavelets_cpu, father_wavelets_cpu = learnable_mmf_train(
            A.clone(), L, K, drop, dim, wavelet_indices, rest_indices,
            epochs=epochs, learning_rate=learning_rate, early_stop=False, opt='original'
        )
    time_cpu = time.time() - start_cpu
    
    loss_cpu = torch.norm(A - A_rec_cpu, p='fro').item()
    
    print(f"CPU version completed in {time_cpu:.2f} seconds")
    print(f"CPU final loss: {loss_cpu:.6f}")
    
    # Run CUDA version
    print("\n" + "-" * 80)
    print("Running CUDA version...")
    print("-" * 80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    start_cuda = time.time()
    A_rec_cuda, U_cuda, D_cuda, mother_coeffs_cuda, father_coeffs_cuda, \
        mother_wavelets_cuda, father_wavelets_cuda = learnable_mmf_train_cuda(
            A.clone(), L, K, drop, dim, wavelet_indices, rest_indices,
            epochs=epochs, learning_rate=learning_rate, early_stop=False, opt='original'
        )
    
    # Synchronize CUDA before timing
    torch.cuda.synchronize()
    time_cuda = time.time() - start_cuda
    
    loss_cuda = torch.norm(A.cuda() - A_rec_cuda, p='fro').item()
    
    print(f"CUDA version completed in {time_cuda:.2f} seconds")
    print(f"CUDA final loss: {loss_cuda:.6f}")
    print(f"Speedup: {time_cpu / time_cuda:.2f}x")
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)
    
    all_match = True
    
    # Compare each output
    all_match &= compare_tensors(A_rec_cpu, A_rec_cuda, "A_rec (reconstructed matrix)")
    all_match &= compare_tensors(U_cpu, U_cuda, "U (rotation matrix)")
    all_match &= compare_tensors(D_cpu, D_cuda, "D (diagonal matrix)")
    all_match &= compare_tensors(mother_coeffs_cpu, mother_coeffs_cuda, "Mother coefficients")
    all_match &= compare_tensors(father_coeffs_cpu, father_coeffs_cuda, "Father coefficients")
    all_match &= compare_tensors(mother_wavelets_cpu, mother_wavelets_cuda, "Mother wavelets")
    all_match &= compare_tensors(father_wavelets_cpu, father_wavelets_cuda, "Father wavelets")
    
    # Compare final losses
    loss_diff = abs(loss_cpu - loss_cuda)
    loss_rel_diff = loss_diff / (loss_cpu + 1e-10)
    
    print(f"\nFinal loss comparison:")
    print(f"  CPU loss:  {loss_cpu:.10f}")
    print(f"  CUDA loss: {loss_cuda:.10f}")
    print(f"  Difference: {loss_diff:.2e} (relative: {loss_rel_diff:.2e})")
    
    if loss_rel_diff < 1e-5:
        print(f"  ✓ Losses match")
    else:
        print(f"  ✗ Losses differ")
        all_match = False
    
    # Final verdict
    print("\n" + "=" * 80)
    if all_match:
        print("✓✓✓ VERIFICATION PASSED ✓✓✓")
        print("CPU and CUDA versions produce identical results!")
    else:
        print("✗✗✗ VERIFICATION FAILED ✗✗✗")
        print("CPU and CUDA versions produce different results!")
    print("=" * 80)
    
    return all_match


def main():
    """Main function."""
    # Small test
    print("Test 1: Small matrix (N=50)")
    success1 = run_verification(N=50, L=40, K=2, epochs=100)
    
    print("\n\n")
    
    # Medium test
    print("Test 2: Medium matrix (N=100)")
    success2 = run_verification(N=100, L=90, K=2, epochs=50)
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Test 1 (N=50):  {'PASSED ✓' if success1 else 'FAILED ✗'}")
    print(f"Test 2 (N=100): {'PASSED ✓' if success2 else 'FAILED ✗'}")
    
    if success1 and success2:
        print("\nAll tests passed! CUDA version is verified. ✓✓✓")
    else:
        print("\nSome tests failed! Please investigate. ✗✗✗")
    print("=" * 80)


if __name__ == '__main__':
    main()