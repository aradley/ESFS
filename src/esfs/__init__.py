"""
ESFS - Entropy Sorting Feature Selection
"""

__version__ = "2.0.0"

# Backend configuration
from . import backend as _backend_module
from .backend import use_cpu, use_gpu, use_mlx, configure, get_backend_info

# Core ESFS functions
from .ESFS import (
    create_scaled_matrix,
    parallel_calc_es_matrices,
    ES_CCF,
    ES_FMG,
)

# Plotting and analysis functions
from .plotting import (
    knn_smooth_gene_expression,
    ES_rank_genes,
    plot_top_ranked_genes_UMAP,
    get_gene_cluster_cell_UMAPs,
    plot_gene_cluster_cell_UMAPs,
)

# Re-assign backend module after imports
backend = _backend_module
del _backend_module

# Define public API
__all__ = [
    # Version
    "__version__",
    # Backend
    "backend",
    "use_cpu",
    "use_gpu",
    "use_mlx",
    "configure",
    "get_backend_info",
    # Core functions
    "create_scaled_matrix",
    "parallel_calc_es_matrices",
    "ES_CCF",
    "ES_FMG",
    # Plotting/analysis
    "knn_smooth_gene_expression",
    "ES_rank_genes",
    "plot_top_ranked_genes_UMAP",
    "get_gene_cluster_cell_UMAPs",
    "plot_gene_cluster_cell_UMAPs",
]
