"""
Test ES_CCF MLX kernel on a subset (50 genes) of the large Roome dataset.
"""
import time
import numpy as np
import scanpy as sc
import sys
sys.path.insert(0, "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Claude_Repo/ESFS/src")

# Load the large dataset
data_path = "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Roome_Data/Compiled_Roome.h5ad"
print("Loading large test data...")
adata_full = sc.read_h5ad(data_path)
print(f"Full data shape: {adata_full.shape}")
print(f"Number of cells: {adata_full.n_obs:,}")

# Subset to first 200 genes for speed test
n_test_genes = 200
print(f"\nSubsetting to first {n_test_genes} genes for speed test...")
adata = adata_full[:, :n_test_genes].copy()
print(f"Test data shape: {adata.shape}")

# Get secondary features label from full data
ess_keys = [k for k in adata_full.varm.keys() if k.endswith("_ESSs")]
secondary_features_label = ess_keys[0].replace("_ESSs", "")
print(f"Secondary features label: {secondary_features_label}")

# Copy relevant varm data for the subset
for key in adata_full.varm.keys():
    if key.startswith(secondary_features_label):
        varm_data = adata_full.varm[key]
        # Handle both DataFrame and numpy array
        if hasattr(varm_data, 'iloc'):
            adata.varm[key] = varm_data.iloc[:n_test_genes].copy()
        else:
            adata.varm[key] = varm_data[:n_test_genes].copy()

# Copy relevant obsm data
sf_key = secondary_features_label + "_secondary_features"
if sf_key in adata_full.obsm:
    adata.obsm[sf_key] = adata_full.obsm[sf_key]

import esfs
from esfs.backend import backend

# Test CPU
print("\n" + "="*60)
print("=== Test 1: CPU Backend ===")
print("="*60)
esfs.use_cpu()
print(f"Backend: USING_MLX={esfs.ESFS.USING_MLX}")

adata_cpu = adata.copy()
start = time.time()
adata_cpu = esfs.ES_CCF(adata_cpu, secondary_features_label, use_cores=-1)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.2f}s")

# Get CPU results
cpu_results = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
cpu_ess = cpu_results["Max_ESSs"].values
cpu_eps = cpu_results["EPs"].values

# Test MLX
print("\n" + "="*60)
print("=== Test 2: MLX Backend ===")
print("="*60)
esfs.use_mlx()
print(f"Backend: USING_MLX={esfs.ESFS.USING_MLX}")

adata_mlx = adata.copy()
start = time.time()
adata_mlx = esfs.ES_CCF(adata_mlx, secondary_features_label, use_cores=-1)
mlx_time = time.time() - start
print(f"MLX time: {mlx_time:.2f}s")

# Get MLX results
mlx_results = adata_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
mlx_ess = mlx_results["Max_ESSs"].values
mlx_eps = mlx_results["EPs"].values

# Compare
print("\n" + "="*60)
print("=== Comparison ===")
print("="*60)

ess_diff = np.abs(cpu_ess - mlx_ess)
eps_diff = np.abs(cpu_eps - mlx_eps)

print(f"\nESS Differences:")
print(f"  Max diff:  {np.max(ess_diff):.6e}")
print(f"  Mean diff: {np.mean(ess_diff):.6e}")

eps_diff_valid = eps_diff[~np.isnan(eps_diff)]
if len(eps_diff_valid) > 0:
    print(f"\nEP Differences:")
    print(f"  Max diff:  {np.max(eps_diff_valid):.6e}")
    print(f"  Mean diff: {np.mean(eps_diff_valid):.6e}")

print("\n" + "="*60)
print("=== Summary ===")
print("="*60)
print(f"CPU time ({n_test_genes} genes, {adata.n_obs:,} cells): {cpu_time:.2f}s")
print(f"MLX time ({n_test_genes} genes, {adata.n_obs:,} cells): {mlx_time:.2f}s")
print(f"Speedup: {cpu_time/mlx_time:.2f}x")

# Estimate for full dataset
full_genes = adata_full.n_vars
est_cpu_time = cpu_time * (full_genes / n_test_genes)
est_mlx_time = mlx_time * (full_genes / n_test_genes)
print(f"\nEstimated time for full dataset ({full_genes:,} genes):")
print(f"  CPU: {est_cpu_time:.0f}s ({est_cpu_time/60:.1f} min)")
print(f"  MLX: {est_mlx_time:.0f}s ({est_mlx_time/60:.1f} min)")
