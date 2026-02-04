"""
Debug script to find where CPU and MLX paths diverge in ES_CCF.
"""
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

import esfs
from esfs.backend import backend

# Run CPU version and capture intermediate results
esfs.use_cpu()
print("\n=== Running CPU version ===")
adata_cpu = adata.copy()
adata_cpu = esfs.ES_CCF(adata_cpu, secondary_features_label, use_cores=-1)

# Run MLX version and capture intermediate results
esfs.use_mlx()
print("\n=== Running MLX version ===")
adata_mlx = adata.copy()
adata_mlx = esfs.ES_CCF(adata_mlx, secondary_features_label, use_cores=-1)

# Compare the Max_Combinatorial_ESSs results
cpu_results = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
mlx_results = adata_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]

cpu_ess = cpu_results["Max_ESSs"].values
mlx_ess = mlx_results["Max_ESSs"].values
cpu_eps = cpu_results["EPs"].values
mlx_eps = mlx_results["EPs"].values

print("\n=== Detailed Comparison ===")
print(f"Total genes: {len(cpu_ess)}")

# Find genes where ESS differs significantly
ess_diff = np.abs(cpu_ess - mlx_ess)
eps_diff = np.abs(cpu_eps - mlx_eps)

print(f"\nESS differences:")
print(f"  Max diff: {np.max(ess_diff):.6e}")
print(f"  Mean diff: {np.mean(ess_diff):.6e}")
print(f"  Genes with diff > 1e-5: {np.sum(ess_diff > 1e-5)}")

print(f"\nEP differences:")
print(f"  Max diff: {np.max(eps_diff):.6e}")
print(f"  Mean diff: {np.mean(eps_diff):.6e}")
print(f"  Genes with diff > 0.01: {np.sum(eps_diff > 0.01)}")
print(f"  Genes with diff > 0.001: {np.sum(eps_diff > 0.001)}")

# Find the gene(s) with the largest EP difference
worst_idx = np.argmax(eps_diff)
print(f"\n=== Worst gene (index {worst_idx}) ===")
print(f"Gene name: {adata.var_names[worst_idx]}")
print(f"CPU ESS: {cpu_ess[worst_idx]:.6f}, MLX ESS: {mlx_ess[worst_idx]:.6f}")
print(f"CPU EP: {cpu_eps[worst_idx]:.6f}, MLX EP: {mlx_eps[worst_idx]:.6f}")
print(f"EP diff: {eps_diff[worst_idx]:.6f}")

# Check if the issue is related to NaN values
cpu_nan_eps = np.sum(np.isnan(cpu_eps))
mlx_nan_eps = np.sum(np.isnan(mlx_eps))
print(f"\nNaN values in EPs:")
print(f"  CPU: {cpu_nan_eps}")
print(f"  MLX: {mlx_nan_eps}")

# Check where one is NaN and the other isn't
cpu_nan_mask = np.isnan(cpu_eps)
mlx_nan_mask = np.isnan(mlx_eps)
mismatched_nan = cpu_nan_mask != mlx_nan_mask
print(f"  Mismatched NaN positions: {np.sum(mismatched_nan)}")

if np.sum(mismatched_nan) > 0:
    print("\n=== Genes with mismatched NaN ===")
    mismatch_indices = np.where(mismatched_nan)[0][:5]  # First 5
    for idx in mismatch_indices:
        print(f"  Gene {idx} ({adata.var_names[idx]}): CPU EP={cpu_eps[idx]:.6f}, MLX EP={mlx_eps[idx]:.6f}")

# Compare the Max_ESS_Features (which clusters are selected)
cpu_features = adata_cpu.obsm[f"{secondary_features_label}_Max_ESS_Features"]
mlx_features = adata_mlx.obsm[f"{secondary_features_label}_Max_ESS_Features"]

# Convert to dense for comparison
if hasattr(cpu_features, 'toarray'):
    cpu_features = cpu_features.toarray()
if hasattr(mlx_features, 'toarray'):
    mlx_features = mlx_features.toarray()

feature_diff = np.abs(cpu_features - mlx_features)
print(f"\n=== Max_ESS_Features comparison ===")
print(f"Max diff: {np.max(feature_diff)}")
print(f"Non-zero diffs: {np.sum(feature_diff > 0)}")

# Check if different clusters are being selected
genes_with_diff_clusters = np.sum(np.any(feature_diff > 0, axis=0))
print(f"Genes with different cluster selections: {genes_with_diff_clusters}")
