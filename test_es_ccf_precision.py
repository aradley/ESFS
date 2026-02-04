"""
Test to verify if EP differences are due to float32 vs float64 precision.
"""
import time
import numpy as np
import scanpy as sc
import sys
sys.path.insert(0, "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Claude_Repo/ESFS/src")

# Load data
data_path = "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/ESFS_Paper/Paper_Submission/Data/Peri_implantation_Human_Embryo_Example/ES_adata.h5ad"
print("Loading data...")
adata = sc.read_h5ad(data_path)
print(f"Data shape: {adata.shape}")

# Identify the secondary_features_label
ess_keys = [k for k in adata.varm.keys() if k.endswith("_ESSs")]
secondary_features_label = ess_keys[0].replace("_ESSs", "")
print(f"Using secondary_features_label: {secondary_features_label}")

# Import esfs and backend
import esfs
from esfs.backend import backend

# Test 1: CPU Backend (float64 default)
esfs.use_cpu()
print("\n" + "="*60)
print("=== Test 1: CPU Backend (float64) ===")
print("="*60)
print(f"Backend dtype: {backend.dtype}")
adata_cpu = adata.copy()
start = time.time()
adata_cpu = esfs.ES_CCF(adata_cpu, secondary_features_label, use_cores=-1)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.2f}s")

# Test 2: MLX Backend with float32 (default)
esfs.configure(gpu=True, upcast=False, verbose=False)
print("\n" + "="*60)
print("=== Test 2: MLX Backend (float32) ===")
print("="*60)
print(f"Backend dtype: {backend.dtype}")
adata_mlx32 = adata.copy()
start = time.time()
adata_mlx32 = esfs.ES_CCF(adata_mlx32, secondary_features_label, use_cores=-1)
mlx32_time = time.time() - start
print(f"MLX float32 time: {mlx32_time:.2f}s")

# Test 3: MLX Backend with float64 (upcast)
esfs.configure(gpu=True, upcast=True, verbose=False)
print("\n" + "="*60)
print("=== Test 3: MLX Backend (float64 upcast) ===")
print("="*60)
print(f"Backend dtype: {backend.dtype}")
adata_mlx64 = adata.copy()
start = time.time()
adata_mlx64 = esfs.ES_CCF(adata_mlx64, secondary_features_label, use_cores=-1)
mlx64_time = time.time() - start
print(f"MLX float64 time: {mlx64_time:.2f}s")

# Compare results
print("\n" + "="*60)
print("=== Comparing Results ===")
print("="*60)

cpu_max_ess = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
cpu_eps = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

mlx32_max_ess = adata_mlx32.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
mlx32_eps = adata_mlx32.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

mlx64_max_ess = adata_mlx64.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
mlx64_eps = adata_mlx64.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

print("\n--- CPU vs MLX float32 ---")
print(f"Max ESS diff: {np.max(np.abs(cpu_max_ess - mlx32_max_ess)):.6e}")
print(f"Max EP diff:  {np.max(np.abs(cpu_eps - mlx32_eps)):.6e}")
print(f"Mean EP diff: {np.mean(np.abs(cpu_eps - mlx32_eps)):.6e}")

print("\n--- CPU vs MLX float64 ---")
print(f"Max ESS diff: {np.max(np.abs(cpu_max_ess - mlx64_max_ess)):.6e}")
print(f"Max EP diff:  {np.max(np.abs(cpu_eps - mlx64_eps)):.6e}")
print(f"Mean EP diff: {np.mean(np.abs(cpu_eps - mlx64_eps)):.6e}")

print("\n--- MLX float32 vs MLX float64 ---")
print(f"Max ESS diff: {np.max(np.abs(mlx32_max_ess - mlx64_max_ess)):.6e}")
print(f"Max EP diff:  {np.max(np.abs(mlx32_eps - mlx64_eps)):.6e}")
print(f"Mean EP diff: {np.mean(np.abs(mlx32_eps - mlx64_eps)):.6e}")

print("\n" + "="*60)
print("=== Summary ===")
print("="*60)
print(f"CPU (float64) time:       {cpu_time:.2f}s")
print(f"MLX float32 time:         {mlx32_time:.2f}s")
print(f"MLX float64 time:         {mlx64_time:.2f}s")

# Check if float64 fixes the precision issue
cpu_mlx64_close = np.allclose(cpu_eps, mlx64_eps, rtol=1e-5, atol=1e-5)
print(f"\nCPU vs MLX64 EPs close (rtol=1e-5): {cpu_mlx64_close}")

if cpu_mlx64_close:
    print("\n✓ CONFIRMED: EP differences are due to float32 vs float64 precision")
else:
    print("\n✗ EP differences NOT fully explained by precision - investigate further")
