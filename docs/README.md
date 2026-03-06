# ESFS Documentation

ESFS is a feature selection package built on the **Entropy Sorting (ES)** mathematical framework, designed for feature selection and marker gene identification in single-cell RNA sequencing datasets.

---

## [Installation](installation.md)

How to install ESFS and its optional GPU backends.

- [Standard CPU install](installation.md#standard-install-cpu)
- [NVIDIA/CUDA GPU acceleration](installation.md#gpu-acceleration-nvidiacuda) — including HPC setup notes
- [Apple Silicon/MLX GPU acceleration](installation.md#gpu-acceleration-apple-silicon--mlx)
- [Backend auto-detection](installation.md#backend-auto-detection)

---

## [Getting Started](getting_started.md)

Worked example notebooks demonstrating the full ESFS pipeline on published single-cell datasets.

- [Peri-implantation Human Embryo](getting_started.md#example-workflows)
- [Delile 2019 Mouse Neural Tube](getting_started.md#example-workflows)
- [Paragi 2022 Mouse Colon](getting_started.md#example-workflows)
- [Ravi 2022 Human Glioblastoma](getting_started.md#example-workflows)

---

## [API Reference](api/index.md)

Full documentation for all public functions.

### [Preprocessing](api/preprocessing.md)

| Function | Description |
|----------|-------------|
| [`create_scaled_matrix()`](api/preprocessing.md#create_scaled_matrix) | Scale expression data to [0, 1] and filter low-expression genes |
| [`parallel_calc_es_matrices()`](api/preprocessing.md#parallel_calc_es_matrices) | Calculate pairwise ES metrics (ESS, EP, SW, SG, SD) for all feature pairs — the ES-GSS step |

### [Algorithms](api/algorithms.md)

| Function | Description |
|----------|-------------|
| [`ES_CCF()`](api/algorithms.md#es_ccf) | Entropy Sorting Combinatorial Cluster Finder — find the combination of clusters that maximises ESS correlation for each gene |
| [`ES_FMG()`](api/algorithms.md#es_fmg) | Entropy Sorting Find Marker Genes — select N genes capturing maximally distinct expression patterns |

### [Plotting & Analysis](api/plotting.md)

| Function | Description |
|----------|-------------|
| [`knn_smooth_gene_expression()`](api/plotting.md#knn_smooth_gene_expression) | Smooth gene expression by averaging over k nearest neighbours |
| [`ES_rank_genes()`](api/plotting.md#es_rank_genes) | Rank genes by weighted ESS network connectivity |
| [`plot_top_ranked_genes_UMAP()`](api/plotting.md#plot_top_ranked_genes_umap) | Embed top-ranked genes in UMAP space |
| [`get_gene_cluster_cell_UMAPs()`](api/plotting.md#get_gene_cluster_cell_umaps) | Generate per-gene-cluster cell UMAP embeddings |
| [`plot_gene_cluster_cell_UMAPs()`](api/plotting.md#plot_gene_cluster_cell_umaps) | Visualise cell UMAPs for each gene cluster |
| [`save_umap_model()`](api/plotting.md#save_umap_model) | Save a fitted UMAP model and gene list to disk |
| [`load_umap_model()`](api/plotting.md#load_umap_model) | Load a previously saved UMAP model |

### [Backend Configuration](api/backend.md)

| Function | Description |
|----------|-------------|
| [`use_cpu()`](api/backend.md#use_cpu) | Force CPU backend (NumPy/Numba) |
| [`use_gpu()`](api/backend.md#use_gpu) | Force GPU backend (CUDA/CuPy, with MLX fallback) |
| [`use_mlx()`](api/backend.md#use_mlx) | Force MLX/Metal backend (Apple Silicon) |
| [`configure()`](api/backend.md#configure) | Full backend configuration (device + precision) |
| [`get_backend_info()`](api/backend.md#get_backend_info) | Return a string describing the current backend |
