"""
MLX Metal backend for ESFS GPU acceleration on Apple Silicon.

This module provides Metal GPU acceleration for sparse overlap calculations
using Apple's MLX framework with custom Metal kernels.
"""

import numpy as np

# Lazy import MLX to avoid import errors on non-Mac systems
_mlx_available = None
# Cache kernels by precision
_overlaps_kernel_f32 = None
_overlaps_kernel_f64 = None


def is_mlx_available():
    """Check if MLX is available and Metal GPU is accessible."""
    global _mlx_available
    if _mlx_available is None:
        try:
            import mlx.core as mx
            _mlx_available = mx.metal.is_available()
        except ImportError:
            _mlx_available = False
    return _mlx_available


def get_overlaps_kernel(use_float64=False):
    """Get or create the overlaps Metal kernel with configurable precision.

    Parameters:
        use_float64: If True, use double precision. If False (default), use float32.

    Returns:
        Compiled Metal kernel for the specified precision.
    """
    global _overlaps_kernel_f32, _overlaps_kernel_f64

    # Return cached kernel if available
    if use_float64 and _overlaps_kernel_f64 is not None:
        return _overlaps_kernel_f64
    if not use_float64 and _overlaps_kernel_f32 is not None:
        return _overlaps_kernel_f32

    import mlx.core as mx

    # Select types based on precision
    dtype_metal = "double" if use_float64 else "float"
    suffix = "" if use_float64 else "f"

    source = f"""
    uint2 gid = thread_position_in_grid.xy;
    uint i = gid.x;  // fixed_feature index
    uint j = gid.y;  // feature index

    if (i >= n_fixed_features || j >= n_features) return;

    // Get column ranges for CSC format
    int ff_start = ff_indptr[i];
    int ff_end = ff_indptr[i + 1];
    int gs_start = gs_indptr[j];
    int gs_end = gs_indptr[j + 1];

    // Initialize accumulators
    {dtype_metal} overlap_sum = 0.0{suffix};
    {dtype_metal} inverse_overlap_sum = 0.0{suffix};

    // Two-pointer merge algorithm
    int ff_ptr = ff_start;
    int gs_ptr = gs_start;

    while (ff_ptr < ff_end && gs_ptr < gs_end) {{
        int ff_row = ff_indices[ff_ptr];
        int gs_row = gs_indices[gs_ptr];

        if (ff_row == gs_row) {{
            // Case 1: Both non-zero at this row
            {dtype_metal} ff_val = ff_data[ff_ptr];
            {dtype_metal} gs_val = gs_data[gs_ptr];
            overlap_sum += metal::min(ff_val, gs_val);
            inverse_overlap_sum += metal::min(1.0{suffix} - ff_val, gs_val);
            ff_ptr++;
            gs_ptr++;
        }} else if (ff_row < gs_row) {{
            // Case 2: ff non-zero, gs zero - no contribution to either
            ff_ptr++;
        }} else {{
            // Case 3: ff zero, gs non-zero
            // overlap: min(0, gs_val) = 0, no contribution
            // inverse_overlap: min(1-0, gs_val) = gs_val
            inverse_overlap_sum += gs_data[gs_ptr];
            gs_ptr++;
        }}
    }}

    // Handle remaining gs elements (ff exhausted, so ff=0 for these rows)
    while (gs_ptr < gs_end) {{
        inverse_overlap_sum += gs_data[gs_ptr];
        gs_ptr++;
    }}

    // Store results (row-major layout)
    uint idx = i * n_features + j;
    overlaps[idx] = overlap_sum;
    inverse_overlaps[idx] = inverse_overlap_sum;
    """

    kernel = mx.fast.metal_kernel(
        name=f"overlaps_and_inverse_{'f64' if use_float64 else 'f32'}",
        input_names=["ff_data", "ff_indices", "ff_indptr",
                     "gs_data", "gs_indices", "gs_indptr",
                     "n_fixed_features", "n_features"],
        output_names=["overlaps", "inverse_overlaps"],
        source=source,
    )

    # Cache the kernel
    if use_float64:
        _overlaps_kernel_f64 = kernel
    else:
        _overlaps_kernel_f32 = kernel

    return kernel


def overlaps_and_inverse_mlx(fixed_features_csc, global_scaled_matrix_csc, use_float64=False):
    """
    Compute overlaps and inverse_overlaps using MLX Metal kernel.

    This function mirrors the CPU overlaps_and_inverse_sparse() function,
    computing both outputs in a single pass for efficiency.

    Parameters:
        fixed_features_csc: scipy.sparse CSC matrix of fixed features
        global_scaled_matrix_csc: scipy.sparse CSC matrix of global scaled data
        use_float64: If True, use double precision. If False (default), use float32.
                     Note: Metal GPUs don't support float64, so this will raise an error.

    Returns:
        (overlaps, inverse_overlaps): Both numpy arrays of shape
            (n_fixed_features, n_features)
    """
    import mlx.core as mx

    # Metal GPUs don't support float64 - raise informative error
    if use_float64:
        raise ValueError(
            "float64 is not supported on Metal GPUs (Apple Silicon). "
            "Use backend.configure(gpu=True, upcast=False) for float32, "
            "or use backend.use_cpu() to force CPU for float64 precision."
        )

    # Use float32 (the only supported precision on Metal)
    mx_dtype = mx.float32
    np_dtype = np.float32

    n_fixed_features = fixed_features_csc.shape[1]
    n_features = global_scaled_matrix_csc.shape[1]

    # Convert scipy sparse CSC components to MLX arrays
    ff_data = mx.array(fixed_features_csc.data.astype(np_dtype))
    ff_indices = mx.array(fixed_features_csc.indices.astype(np.int32))
    ff_indptr = mx.array(fixed_features_csc.indptr.astype(np.int32))

    gs_data = mx.array(global_scaled_matrix_csc.data.astype(np_dtype))
    gs_indices = mx.array(global_scaled_matrix_csc.indices.astype(np.int32))
    gs_indptr = mx.array(global_scaled_matrix_csc.indptr.astype(np.int32))

    # Get or create the kernel for this precision
    kernel = get_overlaps_kernel(use_float64=use_float64)

    # Use 16x16 threadgroups (256 threads per group)
    threadgroup_size = (16, 16, 1)

    # Execute kernel
    outputs = kernel(
        inputs=[ff_data, ff_indices, ff_indptr,
                gs_data, gs_indices, gs_indptr,
                mx.array(n_fixed_features, dtype=mx.int32),
                mx.array(n_features, dtype=mx.int32)],
        grid=(n_fixed_features, n_features, 1),
        threadgroup=threadgroup_size,
        output_shapes=[(n_fixed_features, n_features),
                       (n_fixed_features, n_features)],
        output_dtypes=[mx_dtype, mx_dtype],
    )

    # Convert back to numpy
    return np.array(outputs[0]), np.array(outputs[1])


# Cache for ES_CCF overlap kernel
_esccf_overlaps_kernel_f32 = None


def get_esccf_overlaps_kernel():
    """Get or create the ES_CCF overlaps Metal kernel.

    This kernel computes overlaps with per-feature sort orders and cumulative sums.
    Uses a 1D grid with one thread per gene, processing all cumsum levels incrementally.
    Binary search is used for sparse lookups.

    Returns:
        Compiled Metal kernel for float32 precision.
    """
    global _esccf_overlaps_kernel_f32

    if _esccf_overlaps_kernel_f32 is not None:
        return _esccf_overlaps_kernel_f32

    import mlx.core as mx

    # Metal kernel for ES_CCF overlap computation
    # Grid: (n_fixed_features,) - one thread per gene
    # Each thread processes all cumsum levels sequentially with binary search
    source = """
    uint ff_idx = thread_position_in_grid.x;
    if (ff_idx >= n_fixed_features) return;

    // Get fixed feature column range (CSC format)
    int ff_start = ff_indptr[ff_idx];
    int ff_end = ff_indptr[ff_idx + 1];
    int ff_nnz = ff_end - ff_start;

    // Track total active samples across cumulative clusters
    int total_active = 0;

    // Process each cumsum level incrementally
    for (int cumsum_idx = 0; cumsum_idx < (int)n_clusters; cumsum_idx++) {
        // Get the cluster to add at this level
        int cluster_idx = sort_orders[ff_idx * n_clusters + cumsum_idx];
        int sf_cl_start = sf_indptr[cluster_idx];
        int sf_cl_end = sf_indptr[cluster_idx + 1];
        int cluster_size = sf_cl_end - sf_cl_start;

        // Add cluster size to total active
        total_active += cluster_size;

        // Count active samples that have non-zero ff values and compute overlap
        float overlap_sum = 0.0f;
        float inverse_overlap_nonzero_sum = 0.0f;
        int n_active_with_nonzero_ff = 0;

        // For each non-zero entry in fixed feature
        for (int ff_ptr = ff_start; ff_ptr < ff_end; ff_ptr++) {
            int sample = ff_indices[ff_ptr];
            float ff_val = ff_data[ff_ptr];

            // Check if sample is in any cluster 0..cumsum_idx using binary search
            bool in_active = false;
            for (int k = 0; k <= cumsum_idx && !in_active; k++) {
                int cl_idx = sort_orders[ff_idx * n_clusters + k];
                int cl_start = sf_indptr[cl_idx];
                int cl_end = sf_indptr[cl_idx + 1];

                // Binary search for sample in cluster
                int lo = cl_start, hi = cl_end;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (sf_indices[mid] < sample) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if (lo < cl_end && sf_indices[lo] == sample) {
                    in_active = true;
                }
            }

            if (in_active) {
                // For one-hot clusters, sf_cumsum = 1 when active
                overlap_sum += metal::min(ff_val, 1.0f);
                inverse_overlap_nonzero_sum += metal::min(1.0f - ff_val, 1.0f);
                n_active_with_nonzero_ff++;
            }
        }

        // Samples in active set with ff=0 contribute 1.0 to inverse_overlap
        int n_active_with_zero_ff = total_active - n_active_with_nonzero_ff;
        float inverse_overlap_total = inverse_overlap_nonzero_sum + (float)n_active_with_zero_ff;

        // Store results for this cumsum level
        uint out_idx = ff_idx * n_clusters + cumsum_idx;
        overlaps[out_idx] = overlap_sum;
        inverse_overlaps[out_idx] = inverse_overlap_total;
    }
    """

    kernel = mx.fast.metal_kernel(
        name="esccf_overlaps_and_inverse_f32",
        input_names=["ff_data", "ff_indices", "ff_indptr",
                     "sf_data", "sf_indices", "sf_indptr",
                     "sort_orders",
                     "n_fixed_features", "n_clusters"],
        output_names=["overlaps", "inverse_overlaps"],
        source=source,
    )

    _esccf_overlaps_kernel_f32 = kernel
    return kernel


def identify_max_ESSs_overlaps_mlx(
    fixed_features,
    secondary_features,
    sort_orders,
    sample_cardinality,
    n_fixed_features,
    n_clusters,
    use_float64=False,
):
    """
    MLX Metal kernel for ES_CCF overlap computation with per-feature sort orders.

    This function computes min(ff[i], cumsum(sf[:, sort_order[i]])[:, j]) for each
    (i, j) combination, where the cumsum is computed according to each feature's
    unique sort order.

    Parameters:
        fixed_features: scipy.sparse CSC matrix of fixed features (n_samples x n_fixed_features)
        secondary_features: scipy.sparse CSC matrix of secondary features (n_samples x n_clusters)
        sort_orders: numpy array of shape (n_fixed_features, n_clusters) with cluster orderings
        sample_cardinality: number of samples (rows)
        n_fixed_features: number of fixed features (columns in fixed_features)
        n_clusters: number of clusters (should be n_clusters, after removing last)
        use_float64: If True, raises error (Metal doesn't support float64)

    Returns:
        (overlaps, inverse_overlaps): Both numpy arrays of shape (n_fixed_features, n_clusters)
    """
    import mlx.core as mx

    # Metal GPUs don't support float64
    if use_float64:
        raise ValueError(
            "float64 is not supported on Metal GPUs (Apple Silicon). "
            "Use esfs.configure(gpu=True, upcast=False) for float32."
        )

    np_dtype = np.float32
    mx_dtype = mx.float32

    # Convert scipy sparse CSC components to MLX arrays
    ff_csc = fixed_features.tocsc()
    sf_csc = secondary_features.tocsc()

    ff_data = mx.array(ff_csc.data.astype(np_dtype))
    ff_indices = mx.array(ff_csc.indices.astype(np.int32))
    ff_indptr = mx.array(ff_csc.indptr.astype(np.int32))

    sf_data = mx.array(sf_csc.data.astype(np_dtype))
    sf_indices = mx.array(sf_csc.indices.astype(np.int32))
    sf_indptr = mx.array(sf_csc.indptr.astype(np.int32))

    # Sort orders array
    sort_orders_mx = mx.array(sort_orders.astype(np.int32).ravel())

    # Get kernel
    kernel = get_esccf_overlaps_kernel()

    # One thread per gene, 256 threads per group
    threadgroup_size = (256, 1, 1)

    # Execute kernel - 1D grid
    outputs = kernel(
        inputs=[ff_data, ff_indices, ff_indptr,
                sf_data, sf_indices, sf_indptr,
                sort_orders_mx,
                mx.array(n_fixed_features, dtype=mx.int32),
                mx.array(n_clusters, dtype=mx.int32)],
        grid=(n_fixed_features, 1, 1),
        threadgroup=threadgroup_size,
        output_shapes=[(n_fixed_features, n_clusters),
                       (n_fixed_features, n_clusters)],
        output_dtypes=[mx_dtype, mx_dtype],
    )

    # Convert back to numpy
    return np.array(outputs[0]), np.array(outputs[1])
