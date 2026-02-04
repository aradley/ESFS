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

# Check what secondary_features_label is available
print("\nAvailable varm keys:", list(adata.varm.keys()))
print("Available obsm keys:", list(adata.obsm.keys()))
print("Available layers:", list(adata.layers.keys()))

# Identify the secondary_features_label (look for *_ESSs pattern)
ess_keys = [k for k in adata.varm.keys() if k.endswith("_ESSs")]
if ess_keys:
    secondary_features_label = ess_keys[0].replace("_ESSs", "")
    print(f"\nUsing secondary_features_label: {secondary_features_label}")
else:
    print("No ESSs keys found - need to run parallel_calc_es_matrices first")
    sys.exit(1)

# Test 1: CPU Backend
import esfs
esfs.use_cpu()
print("\n" + "="*50)
print("=== Testing CPU Backend ===")
print("="*50)
adata_cpu = adata.copy()
start = time.time()
adata_cpu = esfs.ES_CCF(adata_cpu, secondary_features_label, use_cores=-1)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.2f}s")

# Test 2: MLX Backend
esfs.use_mlx()
print("\n" + "="*50)
print("=== Testing MLX Backend ===")
print("="*50)
adata_mlx = adata.copy()
start = time.time()
adata_mlx = esfs.ES_CCF(adata_mlx, secondary_features_label, use_cores=-1)
mlx_time = time.time() - start
print(f"MLX time: {mlx_time:.2f}s")

# Compare results
print("\n" + "="*50)
print("=== Comparing Results ===")
print("="*50)
cpu_max_ess = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
mlx_max_ess = adata_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values

cpu_eps = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values
mlx_eps = adata_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

max_ess_diff = np.max(np.abs(cpu_max_ess - mlx_max_ess))
eps_diff = np.max(np.abs(cpu_eps - mlx_eps))
mean_ess_diff = np.mean(np.abs(cpu_max_ess - mlx_max_ess))
mean_eps_diff = np.mean(np.abs(cpu_eps - mlx_eps))

print(f"Max ESS difference: {max_ess_diff}")
print(f"Mean ESS difference: {mean_ess_diff}")
print(f"Max EP difference: {eps_diff}")
print(f"Mean EP difference: {mean_eps_diff}")
print(f"Results close (rtol=1e-5, atol=1e-5): {np.allclose(cpu_max_ess, mlx_max_ess, rtol=1e-5, atol=1e-5) and np.allclose(cpu_eps, mlx_eps, rtol=1e-5, atol=1e-5)}")
print(f"Results close (rtol=1e-3, atol=1e-3): {np.allclose(cpu_max_ess, mlx_max_ess, rtol=1e-3, atol=1e-3) and np.allclose(cpu_eps, mlx_eps, rtol=1e-3, atol=1e-3)}")

print("\n" + "="*50)
print("=== Summary ===")
print("="*50)
print(f"CPU time: {cpu_time:.2f}s")
print(f"MLX time: {mlx_time:.2f}s")
if mlx_time > 0:
    print(f"Speedup (CPU/MLX): {cpu_time/mlx_time:.2f}x")
