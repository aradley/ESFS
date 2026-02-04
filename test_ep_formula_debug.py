"""
Debug the EP formula for a specific gene to find where CPU and MLX diverge.
"""
import numpy as np
import scanpy as sc
import sys
sys.path.insert(0, "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Claude_Repo/ESFS/src")

# Load data
data_path = "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/ESFS_Paper/Paper_Submission/Data/Peri_implantation_Human_Embryo_Example/ES_adata.h5ad"
print("Loading data...")
adata = sc.read_h5ad(data_path)

ess_keys = [k for k in adata.varm.keys() if k.endswith("_ESSs")]
secondary_features_label = ess_keys[0].replace("_ESSs", "")

import esfs
from esfs.backend import backend

# Run both versions
esfs.use_cpu()
adata_cpu = adata.copy()
adata_cpu = esfs.ES_CCF(adata_cpu, secondary_features_label, use_cores=-1)

esfs.use_mlx()
adata_mlx = adata.copy()
adata_mlx = esfs.ES_CCF(adata_mlx, secondary_features_label, use_cores=-1)

# Get results for the worst gene
cpu_results = adata_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
mlx_results = adata_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
cpu_eps = cpu_results["EPs"].values
mlx_eps = mlx_results["EPs"].values

# Find genes with largest differences
eps_diff = np.abs(cpu_eps - mlx_eps)
worst_indices = np.argsort(eps_diff)[-10:][::-1]

print("\n=== Top 10 genes with largest EP differences ===")
for idx in worst_indices:
    print(f"Gene {idx} ({adata.var_names[idx]}): CPU EP={cpu_eps[idx]:.6f}, MLX EP={mlx_eps[idx]:.6f}, diff={eps_diff[idx]:.6f}")

# Examine the formula: EP = max(D_EP, O_EP)
# D_EP = ((CE - min_E) / D) - IndEnt
# O_EP = (CE / O) - IndEnt
# IndEnt = ind_E / ind_X (either ind_X_1 or ind_X1 depending on SD)

# Check if the issue is related to specific ESS ranges
print("\n=== ESS distribution of genes with large EP diff ===")
cpu_ess = cpu_results["Max_ESSs"].values
for idx in worst_indices[:5]:
    print(f"Gene {idx}: ESS={cpu_ess[idx]:.6f}, CPU EP={cpu_eps[idx]:.6f}, MLX EP={mlx_eps[idx]:.6f}")

# Check correlation between ESS value and EP difference
print("\n=== ESS vs EP difference analysis ===")
low_ess_mask = cpu_ess < 0.01
mid_ess_mask = (cpu_ess >= 0.01) & (cpu_ess < 0.1)
high_ess_mask = cpu_ess >= 0.1

print(f"Low ESS (<0.01): {np.sum(low_ess_mask)} genes, max EP diff: {np.max(eps_diff[low_ess_mask]):.6f}")
print(f"Mid ESS (0.01-0.1): {np.sum(mid_ess_mask)} genes, max EP diff: {np.max(eps_diff[mid_ess_mask]):.6f}")
print(f"High ESS (>=0.1): {np.sum(high_ess_mask)} genes, max EP diff: {np.max(eps_diff[high_ess_mask]):.6f}")

# Check if differences are related to the sign of EP
print("\n=== EP sign analysis ===")
both_pos = (cpu_eps > 0) & (mlx_eps > 0)
both_neg = (cpu_eps < 0) & (mlx_eps < 0)
diff_sign = (cpu_eps > 0) != (mlx_eps > 0)

print(f"Both positive: {np.sum(both_pos)}, max EP diff: {np.max(eps_diff[both_pos]) if np.any(both_pos) else 'N/A':.6f}")
print(f"Both negative: {np.sum(both_neg)}, max EP diff: {np.max(eps_diff[both_neg]) if np.any(both_neg) else 'N/A':.6f}")
print(f"Different sign: {np.sum(diff_sign)}, count with large diff: {np.sum(diff_sign & (eps_diff > 0.001))}")

# Check for potential division issues
print("\n=== Checking for extreme EP values ===")
cpu_extreme = np.abs(cpu_eps) > 1
mlx_extreme = np.abs(mlx_eps) > 1
print(f"CPU extreme EPs (|EP|>1): {np.sum(cpu_extreme)}")
print(f"MLX extreme EPs (|EP|>1): {np.sum(mlx_extreme)}")

# Sample some genes with extreme EPs
if np.any(cpu_extreme):
    extreme_idx = np.where(cpu_extreme)[0][:3]
    print("\nSample extreme CPU EPs:")
    for idx in extreme_idx:
        print(f"  Gene {idx}: CPU EP={cpu_eps[idx]:.6f}, MLX EP={mlx_eps[idx]:.6f}")

if np.any(mlx_extreme):
    extreme_idx = np.where(mlx_extreme)[0][:3]
    print("\nSample extreme MLX EPs:")
    for idx in extreme_idx:
        print(f"  Gene {idx}: CPU EP={cpu_eps[idx]:.6f}, MLX EP={mlx_eps[idx]:.6f}")
