"""
Test script to verify the new MLX overlap kernel for ES_CCF.
Compares MLX results against CPU (numba) results.
"""
import time
import numpy as np
import scanpy as sc
import sys
sys.path.insert(0, "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Claude_Repo/ESFS/src")

# Load the smaller test dataset first
data_path = "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/ESFS_Paper/Paper_Submission/Data/Peri_implantation_Human_Embryo_Example/ES_adata.h5ad"
print("Loading test data...")
adata = sc.read_h5ad(data_path)
print(f"Data shape: {adata.shape}")

# Get secondary features label
ess_keys = [k for k in adata.varm.keys() if k.endswith("_ESSs")]
secondary_features_label = ess_keys[0].replace("_ESSs", "")
print(f"Secondary features label: {secondary_features_label}")

import esfs
from esfs.backend import backend

# Test 1: CPU Backend
print("\n" + "="*60)
print("=== Test 1: CPU Backend ===")
print("="*60)
esfs.use_cpu()
print(f"Backend: USING_GPU={esfs.ESFS.USING_GPU}, USING_MLX={esfs.ESFS.USING_MLX}")

adata_cpu = adata.copy()
start = time.time()
adata_cpu = esfs.ES_CCF(adata_cpu, secondary_features_label, use_cores=-1)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.2f}s")

# Get CPU results
cpu_results = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
cpu_ess = cpu_results["Max_ESSs"].values
cpu_eps = cpu_results["EPs"].values

# Test 2: MLX Backend
print("\n" + "="*60)
print("=== Test 2: MLX Backend ===")
print("="*60)
esfs.use_mlx()
print(f"Backend: USING_GPU={esfs.ESFS.USING_GPU}, USING_MLX={esfs.ESFS.USING_MLX}")

adata_mlx = adata.copy()
start = time.time()
adata_mlx = esfs.ES_CCF(adata_mlx, secondary_features_label, use_cores=-1)
mlx_time = time.time() - start
print(f"MLX time: {mlx_time:.2f}s")

# Get MLX results
mlx_results = adata_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
mlx_ess = mlx_results["Max_ESSs"].values
mlx_eps = mlx_results["EPs"].values

# Compare results
print("\n" + "="*60)
print("=== Comparison ===")
print("="*60)

ess_diff = np.abs(cpu_ess - mlx_ess)
eps_diff = np.abs(cpu_eps - mlx_eps)

# Handle NaN in EP comparison
eps_diff_valid = eps_diff[~np.isnan(eps_diff)]

print(f"\nESS Differences:")
print(f"  Max diff:  {np.max(ess_diff):.6e}")
print(f"  Mean diff: {np.mean(ess_diff):.6e}")
print(f"  Genes with diff > 1e-5: {np.sum(ess_diff > 1e-5)}")

print(f"\nEP Differences (excluding NaN):")
if len(eps_diff_valid) > 0:
    print(f"  Max diff:  {np.max(eps_diff_valid):.6e}")
    print(f"  Mean diff: {np.mean(eps_diff_valid):.6e}")
    print(f"  Genes with diff > 0.01: {np.sum(eps_diff_valid > 0.01)}")
else:
    print("  All EP values are NaN")

# Check for NaN mismatches
cpu_nan = np.isnan(cpu_eps)
mlx_nan = np.isnan(mlx_eps)
nan_mismatch = np.sum(cpu_nan != mlx_nan)
print(f"\nNaN mismatch count: {nan_mismatch}")

# Summary
print("\n" + "="*60)
print("=== Summary ===")
print("="*60)
print(f"CPU time:  {cpu_time:.2f}s")
print(f"MLX time:  {mlx_time:.2f}s")
print(f"Speedup:   {cpu_time/mlx_time:.2f}x")

# Check if results are close enough
ess_close = np.allclose(cpu_ess, mlx_ess, rtol=1e-4, atol=1e-5)
eps_close = np.allclose(cpu_eps, mlx_eps, rtol=1e-3, atol=1e-3, equal_nan=True)

print(f"\nESS values close (rtol=1e-4): {ess_close}")
print(f"EP values close (rtol=1e-3):  {eps_close}")

if ess_close and eps_close:
    print("\n✓ MLX kernel produces results consistent with CPU")
else:
    print("\n✗ Results differ - investigate further")
    # Show some examples of differences
    if not ess_close:
        worst_ess = np.argmax(ess_diff)
        print(f"\nWorst ESS diff at gene {worst_ess}:")
        print(f"  CPU: {cpu_ess[worst_ess]:.6f}, MLX: {mlx_ess[worst_ess]:.6f}")
    if not eps_close:
        worst_ep = np.nanargmax(eps_diff)
        print(f"\nWorst EP diff at gene {worst_ep}:")
        print(f"  CPU: {cpu_eps[worst_ep]:.6f}, MLX: {mlx_eps[worst_ep]:.6f}")
