"""
Test ES_CCF MLX kernel on the large Roome dataset (607K cells).
"""
import time
import numpy as np
import scanpy as sc
import sys
sys.path.insert(0, "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Claude_Repo/ESFS/src")

# Load the large dataset
data_path = "/Users/radleya/The Francis Crick Dropbox/BriscoeJ/Radleya/Roome_Data/Compiled_Roome.h5ad"
print("Loading large test data...")
adata = sc.read_h5ad(data_path)
print(f"Data shape: {adata.shape}")
print(f"Number of cells: {adata.n_obs:,}")
print(f"Number of genes: {adata.n_vars:,}")

# Get secondary features label
ess_keys = [k for k in adata.varm.keys() if k.endswith("_ESSs")]
secondary_features_label = ess_keys[0].replace("_ESSs", "")
print(f"Secondary features label: {secondary_features_label}")

ess_df = adata.varm[ess_keys[0]]
print(f"Number of clusters: {ess_df.shape[1]}")

import esfs
from esfs.backend import backend

print("\n" + "="*60)
print("=== Testing MLX Backend on Large Dataset ===")
print("="*60)

esfs.use_mlx()
print(f"Backend: USING_GPU={esfs.ESFS.USING_GPU}, USING_MLX={esfs.ESFS.USING_MLX}")

adata_mlx = adata.copy()
print("\nStarting ES_CCF (this may take a while)...")
start = time.time()
try:
    adata_mlx = esfs.ES_CCF(adata_mlx, secondary_features_label, use_cores=-1)
    mlx_time = time.time() - start
    print(f"\nMLX time: {mlx_time:.2f}s")

    # Get results
    mlx_results = adata_mlx.varm[f"{secondary_features_label}_Max_Combinatorial_ESSs"]
    mlx_ess = mlx_results["Max_ESSs"].values
    mlx_eps = mlx_results["EPs"].values

    print(f"\nResults summary:")
    print(f"  ESS range: [{np.nanmin(mlx_ess):.4f}, {np.nanmax(mlx_ess):.4f}]")
    print(f"  EP range:  [{np.nanmin(mlx_eps):.4f}, {np.nanmax(mlx_eps):.4f}]")
    print(f"  NaN EPs: {np.sum(np.isnan(mlx_eps))}")

    print("\n✓ MLX ES_CCF completed successfully on large dataset!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
