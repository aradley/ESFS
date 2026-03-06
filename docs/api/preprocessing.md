# Preprocessing

These functions prepare the data for Entropy Sorting (ES) metric calculations.

---

## `create_scaled_matrix`

Scale expression data to [0, 1] per gene and filter out low-expression genes.

This is a required preprocessing step before running [`parallel_calc_es_matrices()`](#parallel_calc_es_matrices). Values are clipped at a user-defined percentile threshold to reduce the influence of outliers, then scaled so that each gene's maximum expression is 1. The result is stored as float32 to save memory.

```python
adata = esfs.create_scaled_matrix(adata)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with raw expression counts in `adata.X` |
| `clip_percentile` | `float` | `97.5` | Per-gene percentile used as the upper clip threshold. Values above this are clipped before scaling. |
| `log_scale` | `bool` | `False` | If `True`, apply log2(x + 1) transformation before scaling. |
| `Min_Total_Expression` | `int` | `50` | Minimum number of non-zero cells for a gene to be retained. Genes below this threshold are removed. |

### Returns

`AnnData` — the input object with `adata.layers["Scaled_Counts"]` added (float32 sparse matrix, values in [0, 1]).

Genes with fewer than `Min_Total_Expression` expressing cells are removed from `adata` entirely; the number removed is printed.

### Example

```python
import esfs
import scanpy as sc

adata = sc.read_h5ad("my_data.h5ad")
adata = esfs.create_scaled_matrix(adata, clip_percentile=97.5)
# adata.layers["Scaled_Counts"] now contains the scaled expression matrix
```

---

## `parallel_calc_es_matrices`

Calculate pairwise Entropy Sorting (ES) metrics for all feature pairs. This is the **ES-GSS (Entropy Sorting Gene Set Selection)** step.

For each pair of features, four ES metrics are computed:

- **ESS (Entropy Sort Score)** — the primary correlation metric; measures how strongly the expression of one feature predicts the ordering of another
- **EP (Error Potential)** — a statistical significance measure for the ESS
- **SW (Sort Weight)** — a secondary weighting metric
- **SG (Sort Gain)** — a gain metric for the sort relationship
- **SD (Sort Direction)** — indicates the direction (positive/negative) of the sort relationship

By default, all genes in `adata` are compared pairwise against each other (`secondary_features_label="Self"`). Alternatively, you can compare against a separate set of features stored in `adata.obsm`.

Results are saved to `adata.varm` using the label as a key prefix.

```python
adata = esfs.parallel_calc_es_matrices(adata)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with `adata.layers["Scaled_Counts"]` populated |
| `secondary_features_label` | `str` | `"Self"` | Label for the secondary features to compare against. `"Self"` computes pairwise metrics across all genes in `adata`. Otherwise, must point to a matrix in `adata.obsm` with shape `(n_cells, n_secondary_features)`. |
| `save_matrices` | `tuple` | `("ESSs", "EPs")` | Which ES metric matrices to save. Options: `"ESSs"`, `"EPs"`, `"SWs"`, `"SGs"`. |
| `use_cores` | `int` | `-1` | Number of CPU cores to use. `-1` uses all available cores minus one. Only relevant for the CPU backend. |
| `chunksize` | `int` or `None` | `None` | Chunk size for progress bar updates. Default is 5% of total features. |

### Returns

`AnnData` — the input object with ES metric matrices added to `adata.varm`:

| Key | Content |
|-----|---------|
| `adata.varm["{label}_ESSs"]` | Entropy Sort Score matrix `(n_genes, n_secondary_features)` |
| `adata.varm["{label}_EPs"]` | Error Potential matrix `(n_genes, n_secondary_features)` |
| `adata.varm["{label}_SWs"]` | Sort Weight matrix *(only if requested)* |
| `adata.varm["{label}_SGs"]` | Sort Grade matrix *(only if requested; required for ES-CCF)* |

### Notes

- Requires `adata.layers["Scaled_Counts"]` — run `create_scaled_matrix()` first.
- GPU acceleration (CUDA or MLX) is used automatically if available. See [Backend configuration](backend.md).
- ES-CCF requires `SGs` to be calculated. If you plan to run `ES_CCF()`, include `"SGs"` in `save_matrices`:
  ```python
  adata = esfs.parallel_calc_es_matrices(adata, save_matrices=("ESSs", "EPs", "SGs"))
  ```

### Example

```python
import esfs

# Default: pairwise across all genes
adata = esfs.parallel_calc_es_matrices(adata)
# adata.varm["Self_ESSs"] and adata.varm["Self_EPs"] are now populated

# With SGs (needed for ES_CCF)
adata = esfs.parallel_calc_es_matrices(
    adata,
    save_matrices=("ESSs", "EPs", "SGs"),
)

# Against a separate set of cluster one-hot encodings (for ES-CCF)
adata = esfs.parallel_calc_es_matrices(
    adata,
    secondary_features_label="ClusterLabels",
    save_matrices=("ESSs", "EPs", "SGs"),
)
```
