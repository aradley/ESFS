# API Reference

## Glossary

| Acronym | Full name |
|---------|-----------|
| ES | Entropy Sorting |
| ESS | Entropy Sort Score |
| EP | Error Potential |
| ES-GSS | Entropy Sorting Gene Set Selection |
| ES-CCF | Entropy Sorting Combinatorial Cluster Finder |
| ES-FMG | Entropy Sorting Find Marker Genes |

---

## Preprocessing

| Function | Description |
|----------|-------------|
| [`create_scaled_matrix()`](preprocessing.md#create_scaled_matrix) | Scale expression data to [0, 1] and filter low-expression genes |
| [`parallel_calc_es_matrices()`](preprocessing.md#parallel_calc_es_matrices) | Calculate pairwise ES metrics (ESS, EP, SW, SG) for all feature pairs — the ES-GSS step |

## Algorithms

| Function | Description |
|----------|-------------|
| [`ES_CCF()`](algorithms.md#es_ccf) | Entropy Sorting Combinatorial Cluster Finder — identify combinatorial cluster marker genes |
| [`ES_FMG()`](algorithms.md#es_fmg) | Entropy Sorting Find Marker Genes — select N genes that maximally capture distinct expression patterns |

## Plotting & Analysis

| Function | Description |
|----------|-------------|
| [`knn_smooth_gene_expression()`](plotting.md#knn_smooth_gene_expression) | Smooth gene expression by averaging over k nearest neighbours |
| [`ES_rank_genes()`](plotting.md#es_rank_genes) | Rank genes by weighted ESS network connectivity |
| [`plot_top_ranked_genes_UMAP()`](plotting.md#plot_top_ranked_genes_umap) | Embed top-ranked genes in UMAP space |
| [`get_gene_cluster_cell_UMAPs()`](plotting.md#get_gene_cluster_cell_umaps) | Generate per-gene-cluster cell UMAP embeddings |
| [`plot_gene_cluster_cell_UMAPs()`](plotting.md#plot_gene_cluster_cell_umaps) | Visualise cell UMAPs for each gene cluster |
| [`save_umap_model()`](plotting.md#save_umap_model) | Save a fitted UMAP model and gene list to disk |
| [`load_umap_model()`](plotting.md#load_umap_model) | Load a previously saved UMAP model |

## Backend configuration

| Function | Description |
|----------|-------------|
| [`use_cpu()`](backend.md#use_cpu) | Force CPU backend (NumPy/Numba) |
| [`use_gpu()`](backend.md#use_gpu) | Force GPU backend (CUDA/CuPy, with MLX fallback) |
| [`use_mlx()`](backend.md#use_mlx) | Force MLX/Metal backend (Apple Silicon) |
| [`configure()`](backend.md#configure) | Full backend configuration (device + precision) |
| [`get_backend_info()`](backend.md#get_backend_info) | Return a string describing the current backend |
