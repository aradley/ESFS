# Getting Started

The best way to learn ESFS is to follow the example workflows in the [`Example_Workflows/`](../Example_Workflows/) folder of this repository. Each workflow is a Jupyter notebook demonstrating the full ESFS pipeline on a published single-cell dataset.

## Typical workflow

An ESFS analysis generally follows this sequence:

1. **Preprocess** — scale and filter the expression matrix with `create_scaled_matrix()`
2. **Calculate ES metrics** — run ES-GSS (Entropy Sorting Gene Set Selection) with `parallel_calc_es_matrices()` to compute pairwise Entropy Sort Scores (ESS) and Error Potentials (EP) for all feature pairs
3. **Rank genes** — use `ES_rank_genes()` to build a weighted gene network and rank genes by connectivity
4. **Visualise** — embed top-ranked genes in UMAP space with `plot_top_ranked_genes_UMAP()` and generate per-cluster cell UMAPs with `get_gene_cluster_cell_UMAPs()`
5. *(Optional)* **Identify cluster markers** — use ES-CCF (Entropy Sorting Combinatorial Cluster Finder) with `ES_CCF()` to find, for each gene, the combination of clusters that maximises its ESS correlation
6. *(Optional)* **Find marker gene sets** — use ES-FMG (Entropy Sorting Find Marker Genes) with `ES_FMG()` to select N genes that maximally capture distinct expression patterns

## Example workflows

| Workflow | Dataset | Description |
|----------|---------|-------------|
| [Peri-implantation Human Embryo](../Example_Workflows/Peri_implantation_Human_Embryo_Example/Perimplantation_Human_Embryo_Workflow.ipynb) | Human peri-implantation embryo scRNA-seq | Full ESFS pipeline on human embryogenesis data |
| [Delile 2019 Mouse Neural Tube](../Example_Workflows/Delile2019_Mouse_Neural_Tube/Delile2019_Workflow.ipynb) | Mouse neural tube scRNA-seq (Delile et al. 2019) | Marker gene identification in developing neural tube |
| [Paragi 2022 Mouse Colon](../Example_Workflows/Paragi2022_Mouse_Colon_Example/Paragi022_Mouse_Colon_Workflow.ipynb) | Mouse colon scRNA-seq (Paragi et al. 2022) | Feature selection in intestinal epithelium |
| [Ravi 2022 Human Glioblastoma](../Example_Workflows/Ravi2022_Human_Glioblastoma/Ravi_Analysis_Workflow.ipynb) | Human glioblastoma scRNA-seq (Ravi et al. 2022) | Tumour cell characterisation |

## Datasets

The datasets used in the example workflows can be downloaded from the [figshare repository](https://figshare.com/s/4e445e7fa03cc4ccd289).

---

## Parameter guide

ESFS has three main parameters to tune. The defaults below are good starting points for most datasets.

### `Num_Top_Ranked_Genes` — ES-GSS gene selection

**Function:** `plot_top_ranked_genes_UMAP(adata, Num_Top_Ranked_Genes=3000)`

This controls how many top-ranked genes are passed to the gene clustering step. After clustering, you select the gene cluster whose cell UMAP embedding best captures the biology of interest.

| If your UMAP looks like... | Try... |
|----------------------------|--------|
| Unstructured, no clear groupings | Decrease — fewer, more tightly co-expressed genes |
| Cells cluster primarily by batch/dataset | Decrease — remove noisier genes that carry technical signal |
| Related cell types are merged or indistinct | Increase — include more genes to resolve finer structure |

Start at **3000** and adjust in steps of ~500–1000. You are looking for embeddings that reveal distinct, biologically meaningful cell states with minimal batch separation.

### `N` — number of ES-FMG marker genes

**Function:** `ES_FMG(adata, N=200, secondary_features_label=...)`

N determines how many marker genes to return. This is conceptually similar to choosing the number of PCA components — there is no single correct value, and it depends on the complexity of your dataset.

- Start at **200–400** for typical datasets.
- Increase if important populations seem to be missing from the selected gene set.
- Decrease if many of the returned genes appear redundant or mark the same population.

### `resolution` — ES-FMG redundancy penalisation

**Function:** `ES_FMG(adata, N=200, resolution=1, secondary_features_label=...)`

The `resolution` parameter (between 0 and 1) controls how strongly ES-FMG penalises overlap between the expression profiles of selected marker genes.

| Value | Effect | Best for |
|-------|--------|----------|
| `resolution = 1` | Strongly penalises overlap — selects maximally distinct profiles | Datasets with discrete, non-overlapping cell types |
| `resolution = 0.5–0.7` | Moderate penalisation — allows partial overlap | Mixed datasets with both discrete states and gradients |
| `resolution < 0.5` | Minimal penalisation — captures graded or transitional expression | Developmental trajectories, spatial gradients |

Start at **1** and decrease if you want to capture genes marking overlapping or transitional cell states (e.g. differentiating progenitors, spatial expression gradients).
