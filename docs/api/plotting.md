# Plotting & Analysis

---

## `knn_smooth_gene_expression`

Smooth gene expression by averaging each cell's expression over its k nearest neighbours.

Distances are computed using the **correlation metric** via a Numba-accelerated, chunked BLAS matrix multiplication. The KNN distance matrix is saved to `adata.obsp['correlation_distance_kNN']` and reused on subsequent calls unless `force_recalculate=True`. On CUDA systems with cuML installed, KNN computation is GPU-accelerated.

```python
adata = esfs.knn_smooth_gene_expression(adata, use_genes=top_genes, knn=30)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with expression data in `adata.X` |
| `use_genes` | `list` | — | List of gene names to use for computing nearest-neighbour distances |
| `knn` | `int` | `30` | Number of nearest neighbours to average over |
| `metric` | `str` | `"correlation"` | Distance metric. Currently only `"correlation"` is supported. |
| `log_scale` | `bool` | `False` | If `True`, apply log2(x + 1) transformation before computing distances |
| `chunksize` | `int` or `None` | `None` | Chunk size for progress updates. Default: min(5000, 5% of cells). |
| `force_recalculate` | `bool` | `False` | If `True`, recompute the KNN matrix even if one is already stored in `adata.obsp`. |

### Returns

`AnnData` — input object with `adata.layers["Smoothed_Expression"]` added (dense float32 array, shape `(n_cells, n_genes)`). The KNN distance matrix is also stored in `adata.obsp['correlation_distance_kNN']` for reuse.

### Example

```python
top_genes = list(adata.var_names[adata.var["ES_Rank"] < 500])
adata = esfs.knn_smooth_gene_expression(adata, use_genes=top_genes, knn=30)
```

---

## `ES_rank_genes`

Rank genes by their connectivity in a weighted ESS (Entropy Sort Score) network.

Genes are first filtered by EP (Error Potential) and ESS thresholds, then genes with too few significant connections (`min_edges`) are iteratively pruned from the network. The remaining genes are ranked by a normalised weighted average of their ESS connections.

```python
adata = esfs.ES_rank_genes(adata)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with ES metrics in `adata.varm` |
| `EP_threshold` | `float` | `0.0` | Minimum EP (Error Potential) for an edge to be included in the network. Increase to require more statistically significant connections. |
| `ESS_threshold` | `float` | `0.01` | Minimum ESS (Entropy Sort Score) for an edge to be included. |
| `exclude_genes` | `tuple` or `None` | `None` | Gene names to forcibly exclude from the network. |
| `known_important_genes` | `tuple` or `None` | `None` | If provided, prints a table showing the ranks of these genes. |
| `secondary_features_label` | `str` | `"Self"` | Key prefix for the ES metric matrices in `adata.varm`. |
| `min_edges` | `int` | `5` | Minimum number of significant edges a gene must have to be retained. Genes below this threshold are pruned iteratively. |

### Returns

`AnnData` — input object with two new columns added to `adata.var`:

| Key | Content |
|-----|---------|
| `adata.var["ESFS_Gene_Weights"]` | Normalised weighted network score for each gene |
| `adata.var["ES_Rank"]` | Integer rank (0 = highest ranked) |

### Example

```python
adata = esfs.ES_rank_genes(
    adata,
    EP_threshold=0.0,
    ESS_threshold=0.01,
    min_edges=5,
    known_important_genes=np.array(["SOX2", "PAX6", "NKX2-2"]),
)
# Top 500 ranked genes:
top_genes = adata.var_names[adata.var["ES_Rank"] < 500]
```

---

## `plot_top_ranked_genes_UMAP`

Embed the top N ranked genes in a UMAP using their pairwise ESS values as the distance metric, and optionally cluster them.

Returns the gene names, cluster labels, and 2D embedding coordinates for further analysis.

```python
top_genes, labels, embedding = esfs.plot_top_ranked_genes_UMAP(
    adata, top_ranked_genes=500, clustering=10
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with `adata.var["ES_Rank"]` populated |
| `top_ranked_genes` | `int` | — | Number of top-ranked genes to embed |
| `clustering` | `None` / `int` / `"hdbscan"` | `None` | Clustering method. `None`: no clustering. Integer: KMeans with that many clusters. `"hdbscan"`: automated density-based clustering. |
| `known_important_genes` | `np.ndarray` | `np.array([])` | Genes to highlight on the UMAP with black crosses |
| `UMAP_min_dist` | `float` | `0.1` | UMAP `min_dist` parameter |
| `UMAP_neighbours` | `int` | `20` | UMAP `n_neighbors` parameter |
| `hdbscan_min_cluster_size` | `int` | `50` | Minimum cluster size for HDBSCAN. Only used when `clustering="hdbscan"`. |
| `secondary_features_label` | `str` | `"Self"` | Key prefix for ES metric matrices in `adata.varm` |
| `random_state` | `int` | `42` | Random seed for reproducibility |

### Returns

A tuple of three arrays:

| Element | Shape | Description |
|---------|-------|-------------|
| `top_ESS_genes` | `(N,)` | Gene names of the top N ranked genes |
| `labels` | `(N,)` | Cluster label for each gene (0 if `clustering=None`) |
| `gene_embedding` | `(N, 2)` | 2D UMAP coordinates for each gene |

### Example

```python
top_genes, labels, embedding = esfs.plot_top_ranked_genes_UMAP(
    adata,
    top_ranked_genes=500,
    clustering=10,
    known_important_genes=np.array(["SOX2", "PAX6"]),
)
```

---

## `get_gene_cluster_cell_UMAPs`

Generate per-gene-cluster cell UMAP embeddings. For each cluster of genes identified by `plot_top_ranked_genes_UMAP()`, fit a UMAP of cells using that cluster's genes as features.

Can also accept a custom gene list directly via `specific_genes`, bypassing cluster-based selection.

```python
embeddings, selected_genes = esfs.get_gene_cluster_cell_UMAPs(
    adata,
    gene_clust_labels=labels,
    top_ESS_genes=top_genes,
    n_neighbors=15,
    min_dist=0.01,
    log_transformed=False,
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with expression data |
| `gene_clust_labels` | `np.ndarray` or `None` | `None` | Cluster label for each gene (from `plot_top_ranked_genes_UMAP()`). Required if `specific_genes` is not provided. |
| `top_ESS_genes` | `np.ndarray` or `None` | `None` | Gene names matching `gene_clust_labels`. Required if `specific_genes` is not provided. |
| `n_neighbors` | `int` | — | UMAP `n_neighbors` parameter *(keyword-only)* |
| `min_dist` | `float` | — | UMAP `min_dist` parameter *(keyword-only)* |
| `log_transformed` | `bool` | — | If `True`, apply log2(x + 1) to expression before fitting UMAP *(keyword-only)* |
| `specific_cluster` | `int` / `list` / `None` | `None` | Run only for a specific cluster label or list of labels. Cannot be used with `specific_genes`. |
| `metric` | `str` | `"correlation"` | Distance metric for UMAP |
| `random_state` | `int` or `None` | `None` | Random seed for reproducibility |
| `memory_limit_gb` | `float` | `5.0` | Maximum memory (GB) for the KNN correlation matrix chunk |
| `specific_genes` | `list` / `np.ndarray` / `None` | `None` | Custom gene list, bypassing cluster-based selection. Cannot be used with `specific_cluster`. |
| `return_model` | `bool` | `False` | If `True`, return fitted UMAP model objects instead of raw embeddings. Required for `save_umap_model()`. |

### Returns

A tuple of:

| Element | Description |
|---------|-------------|
| `gene_cluster_embeddings` | List of UMAP embeddings (shape `(n_cells, 2)`) or model objects (if `return_model=True`) |
| `gene_cluster_selected_genes` | List of gene name lists used for each cluster's embedding |

Display labels for each cluster are also stored in `adata.uns['gene_cluster_labels']`.

### Example

```python
# Standard usage — one UMAP per gene cluster
embeddings, selected_genes = esfs.get_gene_cluster_cell_UMAPs(
    adata,
    gene_clust_labels=labels,
    top_ESS_genes=top_genes,
    n_neighbors=15,
    min_dist=0.01,
    log_transformed=False,
)

# Return saveable UMAP models
embeddings, selected_genes = esfs.get_gene_cluster_cell_UMAPs(
    adata,
    gene_clust_labels=labels,
    top_ESS_genes=top_genes,
    n_neighbors=15,
    min_dist=0.01,
    log_transformed=False,
    return_model=True,
)
```

---

## `plot_gene_cluster_cell_UMAPs`

Visualise the per-gene-cluster cell UMAPs generated by `get_gene_cluster_cell_UMAPs()`.

Cells can be coloured by a categorical label from `adata.obs` or by the expression of a gene from `adata.var`.

```python
esfs.plot_gene_cluster_cell_UMAPs(
    adata,
    gene_cluster_embeddings=embeddings,
    gene_cluster_selected_genes=selected_genes,
    cell_label="leiden",
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object (must have `adata.uns['gene_cluster_labels']` set by `get_gene_cluster_cell_UMAPs()`) |
| `gene_cluster_embeddings` | `list` | — | List of UMAP embeddings from `get_gene_cluster_cell_UMAPs()` |
| `gene_cluster_selected_genes` | `list` | — | List of gene lists from `get_gene_cluster_cell_UMAPs()` |
| `cell_label` | `str` | `"None"` | Column name in `adata.obs` (categorical colouring) or gene name in `adata.var` (expression colouring). Pass `"None"` to skip colouring. |
| `ncol` | `int` | `1` | Number of columns in the figure layout |
| `log2_gene_expression` | `bool` | `True` | Apply log2(x + 1) to expression values when colouring by a gene |
| `figsize` | `tuple` | `(18, 10)` | Figure size |
| `marker_size` | `int` | `3` | Scatter point size |
| `sort_by_value` | `bool` | `True` | When colouring by gene expression, plot cells in order of increasing expression so high-expressing cells appear on top |

### Returns

`None` — displays a matplotlib figure.

### Example

```python
# Colour by cluster label
esfs.plot_gene_cluster_cell_UMAPs(
    adata, embeddings, selected_genes, cell_label="leiden", ncol=2
)

# Colour by gene expression
esfs.plot_gene_cluster_cell_UMAPs(
    adata, embeddings, selected_genes, cell_label="SOX2"
)
```

---

## `save_umap_model`

Save a fitted UMAP model and its associated gene list to disk.

The model, gene list, and metadata are bundled into a single `.joblib` file that can be loaded on any machine with `umap-learn`, `numpy`, `scipy`, and `joblib` installed (no GPU or cuML required).

```python
esfs.save_umap_model(model, gene_list=selected_genes[0], filepath="cluster_0.joblib")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `umap.UMAP` | — | A fitted umap-learn UMAP model (returned by `get_gene_cluster_cell_UMAPs(..., return_model=True)`) |
| `gene_list` | `array-like` | — | The genes used to create the embedding |
| `filepath` | `str` or `Path` | — | Output file path (recommended extension: `.joblib`) |
| `log_transformed` | `bool` | `False` | Whether the training data was log2(x + 1) transformed before fitting. Stored as metadata for correct preprocessing on load. |

### Returns

`None` — writes a `.joblib` file to `filepath`.

---

## `load_umap_model`

Load a UMAP model bundle previously saved with `save_umap_model()`.

```python
bundle = esfs.load_umap_model("cluster_0.joblib")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filepath` | `str` or `Path` | — | Path to the saved `.joblib` file |

### Returns

`dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"model"` | `umap.UMAP` | The fitted model (supports `.transform()`) |
| `"gene_list"` | `np.ndarray` | Genes used for the embedding |
| `"log_transformed"` | `bool` | Whether log2(x + 1) was applied to the training data |

### Example

```python
bundle = esfs.load_umap_model("cluster_0.joblib")
model = bundle["model"]
gene_list = bundle["gene_list"]

# Project new data into the same embedding space
new_X = new_adata[:, gene_list].X.toarray()
if bundle["log_transformed"]:
    new_X = np.log2(new_X + 1)
new_embedding = model.transform(new_X)
```
