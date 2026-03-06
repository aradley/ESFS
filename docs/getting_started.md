# Getting Started

The best way to learn ESFS is to follow the example workflows in the [`Example_Workflows/`](../Example_Workflows/) folder of this repository. Each workflow is a Jupyter notebook demonstrating the full ESFS pipeline on a published single-cell dataset.

## Typical workflow

An ESFS analysis generally follows this sequence:

1. **Preprocess** — scale and filter the expression matrix with `create_scaled_matrix()`
2. **Calculate ES metrics** — run ES-GSS (Entropy Sorting Gene Set Selection) with `parallel_calc_es_matrices()` to compute pairwise Entropy Sort Scores (ESS) and Error Potentials (EP) for all feature pairs
3. **Rank genes** — use `ES_rank_genes()` to build a weighted gene network and rank genes by connectivity
4. **Visualise** — embed top-ranked genes in UMAP space with `plot_top_ranked_genes_UMAP()` and generate per-cluster cell UMAPs with `get_gene_cluster_cell_UMAPs()`
5. *(Optional)* **Identify cluster markers** — use ES-CCF (Entropy Sorting Combinatorial Cluster Finder) with `ES_CCF()` to find combinatorial marker genes for cell clusters
6. *(Optional)* **Find marker gene sets** — use ES-FMG (Entropy Sorting Find Marker Genes) with `ES_FMG()` to select N genes that maximally capture distinct expression patterns

## Example workflows

| Workflow | Dataset | Description |
|----------|---------|-------------|
| [Peri-implantation Human Embryo](../Example_Workflows/Peri_implantation_Human_Embryo_Example/Perimplantation_Human_Embryo_Workflow.ipynb) | Human peri-implantation embryo scRNA-seq | Full ESFS pipeline on human embryogenesis data |
| [Delile 2019 Mouse Neural Tube](../Example_Workflows/Delile2019_Mouse_Neural_Tube/Delile2019_Workflow.ipynb) | Mouse neural tube scRNA-seq (Delile et al. 2019) | Marker gene identification in developing neural tube |
| [Paragi 2022 Mouse Colon](../Example_Workflows/Paragi2022_Mouse_Colon_Example/Paragi022_Mouse_Colon_Workflow.ipynb) | Mouse colon scRNA-seq (Paragi et al. 2022) | Feature selection in intestinal epithelium |
| [Ravi 2022 Human Glioblastoma](../Example_Workflows/Ravi2022_Human_Glioblastoma/Ravi_Anlysis_Workflow.ipynb) | Human glioblastoma scRNA-seq (Ravi et al. 2022) | Tumour cell characterisation |

## Datasets

The datasets used in the example workflows can be downloaded from the Mendeley Data repository linked in the manuscript. A small test dataset for quick experimentation is provided in [`Claude_Dev_Folder/Claude_Test_Data.h5ad`](../Claude_Dev_Folder/Claude_Test_Data.h5ad).
