#!/usr/bin/env python3
"""
Diagnostic script for CUDA->CPU switching issue.
Run this on your CUDA cluster to identify where the conversion is failing.

Usage:
    python diagnose_cuda_cpu_switch.py
"""

import sys
import numpy as np
import scipy.sparse as spsparse

print("="*70)
print("CUDA -> CPU Switching Diagnostic")
print("="*70)

# Step 1: Check CuPy availability
print("\n1. Checking CuPy availability...")
try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    print(f"   CuPy version: {cp.__version__}")
    print(f"   CUDA available: {cp.cuda.is_available()}")
    CUPY_AVAILABLE = True
except ImportError as e:
    print(f"   CuPy not available: {e}")
    CUPY_AVAILABLE = False
    sys.exit("Cannot test CUDA->CPU switching without CuPy")

# Step 2: Import esfs and check initial state
print("\n2. Importing esfs...")
import esfs
from esfs.backend import backend
from esfs import ESFS as esfs_module

print(f"   backend.xp: {backend.xp.__name__}")
print(f"   backend.using_gpu: {backend.using_gpu}")
print(f"   ESFS.xp: {esfs_module.xp.__name__}")
print(f"   ESFS.USING_GPU: {esfs_module.USING_GPU}")

# Step 3: Create a small test sparse matrix with CuPy
print("\n3. Creating test CuPy sparse matrix...")
# Create a small scipy sparse matrix first
scipy_sparse = spsparse.random(100, 50, density=0.1, format='csc', dtype=np.float32)
print(f"   scipy sparse type: {type(scipy_sparse)}")
print(f"   scipy sparse.data dtype: {scipy_sparse.data.dtype}")

# Convert to CuPy sparse
cupy_sparse_mat = cupy_sparse.csc_matrix(scipy_sparse)
print(f"   CuPy sparse type: {type(cupy_sparse_mat)}")
print(f"   CuPy sparse.data type: {type(cupy_sparse_mat.data)}")
print(f"   CuPy sparse.data dtype: {cupy_sparse_mat.data.dtype}")

# Step 4: Switch to CPU mode
print("\n4. Switching to CPU mode...")
esfs.use_cpu()
print(f"   backend.xp: {backend.xp.__name__}")
print(f"   backend.using_gpu: {backend.using_gpu}")
print(f"   ESFS.xp: {esfs_module.xp.__name__}")
print(f"   ESFS.USING_GPU: {esfs_module.USING_GPU}")

# Step 5: Test _ensure_numpy on CuPy arrays
print("\n5. Testing _ensure_numpy on CuPy arrays...")
from esfs.ESFS import _ensure_numpy

# Test on CuPy dense array
cupy_dense = cp.array([1.0, 2.0, 3.0])
print(f"   Input (CuPy dense): {type(cupy_dense)}")
result = _ensure_numpy(cupy_dense)
print(f"   Output: {type(result)}, is numpy? {isinstance(result, np.ndarray)}")

# Test on CuPy sparse matrix
print(f"\n   Input (CuPy sparse): {type(cupy_sparse_mat)}")
result = _ensure_numpy(cupy_sparse_mat)
print(f"   Output: {type(result)}, is scipy sparse? {spsparse.issparse(result)}")

# Test on sparse matrix components
print(f"\n   CuPy sparse.data: {type(cupy_sparse_mat.data)}")
data_result = _ensure_numpy(cupy_sparse_mat.data)
print(f"   After _ensure_numpy: {type(data_result)}, is numpy? {isinstance(data_result, np.ndarray)}")

# Step 6: Test _convert_sparse_array on CuPy sparse
print("\n6. Testing _convert_sparse_array on CuPy sparse...")
from esfs.ESFS import _convert_sparse_array

print(f"   Input: {type(cupy_sparse_mat)}")
result = _convert_sparse_array(cupy_sparse_mat)
print(f"   Output: {type(result)}")
print(f"   Is scipy sparse? {spsparse.issparse(result)}")
if spsparse.issparse(result):
    print(f"   result.data type: {type(result.data)}")
    print(f"   result.data dtype: {result.data.dtype}")
    print(f"   result.indices type: {type(result.indices)}")
    print(f"   result.indptr type: {type(result.indptr)}")

# Step 7: Test the Numba function directly
print("\n7. Testing Numba function with converted data...")
from esfs.ESFS import overlaps_and_inverse_sparse

# Convert CuPy sparse to scipy
scipy_result = cupy_sparse_mat.get()
print(f"   cupy_sparse.get() type: {type(scipy_result)}")
print(f"   data dtype: {scipy_result.data.dtype}")
print(f"   indices dtype: {scipy_result.indices.dtype}")
print(f"   indptr dtype: {scipy_result.indptr.dtype}")

# Make sure it's CSC
scipy_result = scipy_result.tocsc()

# Create a second matrix for overlap calculation
scipy_sparse2 = spsparse.random(100, 30, density=0.1, format='csc', dtype=np.float32)

# Ensure correct dtypes
ff_data = np.ascontiguousarray(scipy_result.data, dtype=np.float32)
ff_indices = np.ascontiguousarray(scipy_result.indices, dtype=np.int32)
ff_indptr = np.ascontiguousarray(scipy_result.indptr, dtype=np.int32)
gs_data = np.ascontiguousarray(scipy_sparse2.data, dtype=np.float32)
gs_indices = np.ascontiguousarray(scipy_sparse2.indices, dtype=np.int32)
gs_indptr = np.ascontiguousarray(scipy_sparse2.indptr, dtype=np.int32)

print(f"\n   Prepared arrays:")
print(f"   ff_data: shape={ff_data.shape}, dtype={ff_data.dtype}, contiguous={ff_data.flags['C_CONTIGUOUS']}")
print(f"   ff_indices: shape={ff_indices.shape}, dtype={ff_indices.dtype}")
print(f"   ff_indptr: shape={ff_indptr.shape}, dtype={ff_indptr.dtype}")
print(f"   gs_data: shape={gs_data.shape}, dtype={gs_data.dtype}")

print("\n   Calling overlaps_and_inverse_sparse...")
try:
    overlaps, inverse_overlaps = overlaps_and_inverse_sparse(
        ff_data, ff_indices, ff_indptr,
        gs_data, gs_indices, gs_indptr,
        100,  # n_samples
        50,   # n_fixed_features
        30,   # n_features
        use_float64=False
    )
    print(f"   SUCCESS!")
    print(f"   overlaps shape: {overlaps.shape}")
    print(f"   overlaps dtype: {overlaps.dtype}")
    print(f"   overlaps NaN count: {np.isnan(overlaps).sum()}")
    print(f"   overlaps range: [{overlaps.min():.4f}, {overlaps.max():.4f}]")

    # Check for all-zero results (which would indicate a problem)
    if overlaps.max() == 0:
        print("   WARNING: All overlaps are zero! This indicates a problem.")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

# Step 8: Full integration test
print("\n8. Full integration test with anndata...")
import anndata as ad

# Create a larger test adata (more samples so genes have >50 expressed cells)
# 500 samples, 100 features, 30% density = ~150 non-zero per gene on average
X = spsparse.random(500, 100, density=0.3, format='csc', dtype=np.float32)
adata = ad.AnnData(X=X)
print(f"   Created test adata: {adata.shape}")

# First, run with GPU
print("   Running create_scaled_matrix with GPU backend...")
esfs.use_gpu()
print(f"   backend.using_gpu: {backend.using_gpu}")
adata = esfs.create_scaled_matrix(adata)

scaled_type = type(adata.layers['Scaled_Counts'])
print(f"   Scaled_Counts type: {scaled_type}")
print(f"   Scaled_Counts.data type: {type(adata.layers['Scaled_Counts'].data)}")

# Now switch to CPU
print("\n   Switching to CPU mode...")
esfs.use_cpu()
print(f"   backend.using_gpu: {backend.using_gpu}")
print(f"   ESFS.USING_GPU: {esfs_module.USING_GPU}")

# Check if Scaled_Counts is still CuPy
print(f"\n   After CPU switch, before calc:")
print(f"   Scaled_Counts type: {type(adata.layers['Scaled_Counts'])}")
print(f"   Scaled_Counts.data type: {type(adata.layers['Scaled_Counts'].data)}")

# Run the calculation
print("\n   Running parallel_calc_es_matrices (subset of 30 features for speed)...")
# Use a subset for faster testing
adata_test = adata[:, :30].copy()
adata_test.layers['Scaled_Counts'] = adata.layers['Scaled_Counts'][:, :30]
print(f"   Test adata shape: {adata_test.shape}")
print(f"   Test Scaled_Counts type: {type(adata_test.layers['Scaled_Counts'])}")
try:
    adata_test = esfs.parallel_calc_es_matrices(
        adata_test,
        secondary_features_label="Self",
        save_matrices=np.array(["ESSs", "EPs"]),
        use_cores=2
    )

    ESSs = adata_test.varm['Self_ESSs']
    print(f"\n   Results:")
    print(f"   ESSs shape: {ESSs.shape}")
    print(f"   ESSs NaN count: {np.isnan(ESSs).sum()} / {ESSs.size}")

    all_nan_count = np.all(np.isnan(ESSs), axis=1).sum()
    if all_nan_count > 0:
        print(f"   PROBLEM: Found {all_nan_count} all-NaN rows!")
    else:
        print(f"   SUCCESS: No all-NaN slices!")
        print(f"   ESSs range: [{np.nanmin(ESSs):.4f}, {np.nanmax(ESSs):.4f}]")

except Exception as e:
    print(f"\n   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Diagnostic complete")
print("="*70)
