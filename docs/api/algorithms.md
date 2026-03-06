# Algorithms

---

## `ES_CCF`

**Entropy Sorting Combinatorial Cluster Finder**

Identify the combination of cell clusters that maximally correlates with the expression profile of each gene — without requiring any prior knowledge of marker genes.

ES-CCF takes a set of secondary features derived from cell cluster labels (typically one-hot encoded cluster assignments) and finds which *combination* of clusters best characterises each gene. This turns the intractable combinatorial search into a tractable linear problem by sorting cluster contributions using the SG (Sort Grade) direction metric.

Requires both `ESSs` and `SGs` to have been calculated by `parallel_calc_es_matrices()` for the given `secondary_features_label`.

> **Note:** ES-CCF always runs on CPU (Numba JIT), regardless of the active backend.

```python
adata = esfs.ES_CCF(adata, secondary_features_label="ClusterLabels")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with ES metrics and cluster one-hot encodings |
| `secondary_features_label` | `str` | — | Key for the secondary features in `adata.obsm` (e.g. one-hot encoded cluster labels) and the corresponding ES metrics in `adata.varm` |
| `use_cores` | `int` | `-1` | Number of CPU cores to use. `-1` uses all available minus one. |
| `chunksize` | `int` or `None` | `None` | Chunk size for progress bar updates. |

### Returns

`AnnData` — the input object with two new attributes added:

| Key | Content |
|-----|---------|
| `adata.varm["{label}_Max_Combinatorial_ESSs"]` | DataFrame with columns `Max_ESSs` and `EPs` for each gene's optimal cluster combination |
| `adata.obsm["{label}_Max_ESS_Features"]` | Sparse matrix of the cluster combination that maximises the ESS for each gene |

### Prerequisites

1. `create_scaled_matrix()` — to populate `adata.layers["Scaled_Counts"]`
2. Create one-hot encoded cluster labels and store in `adata.obsm[secondary_features_label]`
3. `parallel_calc_es_matrices()` with `save_matrices=("ESSs", "EPs", "SGs")` using the same `secondary_features_label`

### Example

```python
import esfs
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import pandas as pd

# Assume adata already has cluster labels in adata.obs["leiden"]
# Step 1: One-hot encode cluster labels
clusters = adata.obs["leiden"].values
unique_clusters = np.unique(clusters)
one_hot = np.zeros((adata.n_obs, len(unique_clusters)), dtype=np.float32)
for i, c in enumerate(unique_clusters):
    one_hot[:, i] = (clusters == c).astype(np.float32)
adata.obsm["ClusterLabels"] = sp.csc_matrix(one_hot)

# Step 2: Calculate ES metrics against cluster labels
adata = esfs.parallel_calc_es_matrices(
    adata,
    secondary_features_label="ClusterLabels",
    save_matrices=("ESSs", "EPs", "SGs"),
)

# Step 3: Run ES-CCF
adata = esfs.ES_CCF(adata, secondary_features_label="ClusterLabels")
# Results in adata.varm["ClusterLabels_Max_Combinatorial_ESSs"]
```

---

## `ES_FMG`

**Entropy Sorting Find Marker Genes**

Select N genes that maximally capture distinct expression patterns across the dataset.

ES-FMG uses a simulated annealing approach with reheating to find a set of N genes whose pairwise Entropy Sort Scores (ESS) are simultaneously high (meaning each gene is strongly correlated with others in the set) and mutually distinct (minimising redundancy between selected genes). The `resolution` parameter controls the balance between these two objectives.

Requires ES-CCF to have been run first, as ES-FMG uses the `_Max_ESS_Features` from ES-CCF as the secondary features for ranking.

> **Note:** ES-FMG always runs on CPU (Numba JIT), regardless of the active backend.

```python
chosen_idxs, chosen_genes, pairwise_ESSs = esfs.ES_FMG(
    adata, N=50, secondary_features_label="ClusterLabels"
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `adata` | `AnnData` | — | AnnData object with ES metrics calculated |
| `N` | `int` | — | Number of marker genes to select |
| `secondary_features_label` | `str` | — | Label matching the ES metrics in `adata.varm` (the same label used in `ES_CCF`) |
| `input_genes` | `list` / `tuple` / `None` | `None` | Restrict the search to a specific gene subset. If `None`, all genes in `adata` are considered. |
| `num_reheats` | `int` | `3` | Number of simulated annealing reheats. More reheats improve optimisation quality but increase runtime. |
| `resolution` | `int` | `1` | Controls the trade-off between selecting high-ESS genes and selecting maximally distinct genes. Higher values penalise redundancy more. |
| `use_cores` | `int` | `-1` | Number of CPU cores to use. `-1` uses all available minus one. |

### Returns

A tuple of three arrays:

| Element | Shape | Description |
|---------|-------|-------------|
| `chosen_idxs` | `(N,)` | Indices of the selected genes within `input_genes` (or `adata.var_names` if `input_genes=None`) |
| `chosen_genes` | `(N,)` | Names of the selected genes |
| `pairwise_ESSs` | `(N, N)` | Pairwise ESS matrix for the selected gene set |

### Prerequisites

1. `create_scaled_matrix()`
2. `parallel_calc_es_matrices()` with `save_matrices=("ESSs", "EPs", "SGs")`
3. `ES_CCF()` — to generate `_Max_ESS_Features` used as secondary features

### Example

```python
chosen_idxs, chosen_genes, pairwise_ESSs = esfs.ES_FMG(
    adata,
    N=50,
    secondary_features_label="ClusterLabels",
    num_reheats=3,
    resolution=1,
)

print("Selected marker genes:", chosen_genes)
```
