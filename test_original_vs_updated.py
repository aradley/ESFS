"""
Compare ES_CCF results between original (GitHub) and updated (local) versions.
"""
import time
import numpy as np
import scanpy as sc
import sys
import shutil
import os

# Load data
data_path = "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/ESFS_Paper/Paper_Submission/Data/Peri_implantation_Human_Embryo_Example/ES_adata.h5ad"
print("Loading data...")
adata = sc.read_h5ad(data_path)
print(f"Data shape: {adata.shape}")

# Identify the secondary_features_label
ess_keys = [k for k in adata.varm.keys() if k.endswith("_ESSs")]
secondary_features_label = ess_keys[0].replace("_ESSs", "")
print(f"Using secondary_features_label: {secondary_features_label}")

# Paths
src_path = "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Claude_Repo/ESFS/src"
original_esfs_path = "/tmp/ESFS_original.py"
updated_esfs_path = f"{src_path}/esfs/ESFS.py"
backup_path = "/tmp/ESFS_backup.py"

# Backup the updated version
shutil.copy(updated_esfs_path, backup_path)

print("\n" + "="*60)
print("=== Test 1: Original GitHub Version (CPU) ===")
print("="*60)

# Replace with original version
shutil.copy(original_esfs_path, updated_esfs_path)

# Clear any cached imports
if 'esfs' in sys.modules:
    # Remove all esfs submodules
    mods_to_remove = [k for k in sys.modules if k.startswith('esfs')]
    for mod in mods_to_remove:
        del sys.modules[mod]

sys.path.insert(0, src_path)
import esfs
esfs.use_cpu()
adata_orig_cpu = adata.copy()
start = time.time()
adata_orig_cpu = esfs.ES_CCF(adata_orig_cpu, secondary_features_label, use_cores=-1)
orig_cpu_time = time.time() - start
print(f"Original CPU time: {orig_cpu_time:.2f}s")

# Get original CPU results
orig_cpu_ess = adata_orig_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
orig_cpu_eps = adata_orig_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

print("\n" + "="*60)
print("=== Test 2: Original GitHub Version (MLX) ===")
print("="*60)

# Clear any cached imports
mods_to_remove = [k for k in sys.modules if k.startswith('esfs')]
for mod in mods_to_remove:
    del sys.modules[mod]

import esfs
esfs.use_mlx()
adata_orig_mlx = adata.copy()
start = time.time()
adata_orig_mlx = esfs.ES_CCF(adata_orig_mlx, secondary_features_label, use_cores=-1)
orig_mlx_time = time.time() - start
print(f"Original MLX time: {orig_mlx_time:.2f}s")

# Get original MLX results
orig_mlx_ess = adata_orig_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
orig_mlx_eps = adata_orig_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

print("\n" + "="*60)
print("=== Test 3: Updated Version (CPU) ===")
print("="*60)

# Restore updated version
shutil.copy(backup_path, updated_esfs_path)

# Clear any cached imports
mods_to_remove = [k for k in sys.modules if k.startswith('esfs')]
for mod in mods_to_remove:
    del sys.modules[mod]

import esfs
esfs.use_cpu()
adata_upd_cpu = adata.copy()
start = time.time()
adata_upd_cpu = esfs.ES_CCF(adata_upd_cpu, secondary_features_label, use_cores=-1)
upd_cpu_time = time.time() - start
print(f"Updated CPU time: {upd_cpu_time:.2f}s")

# Get updated CPU results
upd_cpu_ess = adata_upd_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
upd_cpu_eps = adata_upd_cpu.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

print("\n" + "="*60)
print("=== Test 4: Updated Version (MLX) ===")
print("="*60)

# Clear any cached imports
mods_to_remove = [k for k in sys.modules if k.startswith('esfs')]
for mod in mods_to_remove:
    del sys.modules[mod]

import esfs
esfs.use_mlx()
adata_upd_mlx = adata.copy()
start = time.time()
adata_upd_mlx = esfs.ES_CCF(adata_upd_mlx, secondary_features_label, use_cores=-1)
upd_mlx_time = time.time() - start
print(f"Updated MLX time: {upd_mlx_time:.2f}s")

# Get updated MLX results
upd_mlx_ess = adata_upd_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["Max_ESSs"].values
upd_mlx_eps = adata_upd_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]["EPs"].values

# Comparisons
print("\n" + "="*60)
print("=== Comparisons ===")
print("="*60)

def compare(name1, name2, ess1, eps1, ess2, eps2):
    ess_diff = np.abs(ess1 - ess2)
    eps_diff = np.abs(eps1 - eps2)
    # Handle NaN comparisons
    eps_diff_valid = eps_diff[~np.isnan(eps_diff)]
    print(f"\n--- {name1} vs {name2} ---")
    print(f"Max ESS diff: {np.max(ess_diff):.6e}")
    print(f"Mean ESS diff: {np.mean(ess_diff):.6e}")
    if len(eps_diff_valid) > 0:
        print(f"Max EP diff (excl NaN): {np.max(eps_diff_valid):.6e}")
        print(f"Mean EP diff (excl NaN): {np.mean(eps_diff_valid):.6e}")
    else:
        print("EP diff: All NaN")
    # Count NaN differences
    nan1 = np.isnan(eps1)
    nan2 = np.isnan(eps2)
    nan_mismatch = np.sum(nan1 != nan2)
    print(f"NaN mismatch count: {nan_mismatch}")

# Original CPU vs Original MLX (baseline - how different were they before?)
compare("Orig CPU", "Orig MLX", orig_cpu_ess, orig_cpu_eps, orig_mlx_ess, orig_mlx_eps)

# Updated CPU vs Updated MLX (how different are they now?)
compare("Upd CPU", "Upd MLX", upd_cpu_ess, upd_cpu_eps, upd_mlx_ess, upd_mlx_eps)

# Original CPU vs Updated CPU (did we break CPU?)
compare("Orig CPU", "Upd CPU", orig_cpu_ess, orig_cpu_eps, upd_cpu_ess, upd_cpu_eps)

# Original MLX vs Updated MLX (did we break MLX?)
compare("Orig MLX", "Upd MLX", orig_mlx_ess, orig_mlx_eps, upd_mlx_ess, upd_mlx_eps)

print("\n" + "="*60)
print("=== Summary ===")
print("="*60)
print(f"Original CPU time: {orig_cpu_time:.2f}s")
print(f"Original MLX time: {orig_mlx_time:.2f}s")
print(f"Updated CPU time:  {upd_cpu_time:.2f}s")
print(f"Updated MLX time:  {upd_mlx_time:.2f}s")

# Clean up
os.remove(backup_path)
print("\nValidation complete.")
