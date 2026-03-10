# ESFS (Entropy Sorting Feature Selection)

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/aradley/ESFS)

**Version 2.0.0**

ESFS is an Entropy Sorting based feature selection package primarily developed for feature selection and marker gene identification in single cell RNA sequencing datasets.

## Documentation

Full documentation is available in the [`docs/`](docs/) folder:

- [Installation](docs/installation.md)
- [Getting Started](docs/getting_started.md)
- [Understanding ESFS](docs/understanding_esfs.md)
- [API Reference](docs/api/index.md)

## Why ESFS?

Standard scRNA-seq workflows rely on two steps that can introduce computational artefacts:

- **Highly variable gene (HVG) selection** — ranks genes by dispersion alone, missing structured co-expression signals and biological patterns shared across gene sets
- **PCA / batch integration** — compresses data into a latent space that can blend biologically distinct signals and distort true biological relationships

ESFS takes a different approach: it works directly in gene expression space, using information theory to identify genes where biological signal outweighs technical noise — without requiring dimensionality reduction or batch correction.

| Challenge | ESFS solution |
|-----------|---------------|
| HVG selection ranks genes by dispersion alone, missing co-expression structure | **ES-GSS** — selects genes based on pairwise information content across all gene pairs |
| Fixing a clustering resolution forces a trade-off between over- and under-partitioning | **ES-CCF** — finds the optimal combination of clusters for each individual gene |
| Differential expression analysis is tied to discrete cluster boundaries | **ES-FMG** — selects N marker genes capturing distinct, non-redundant expression profiles |

For a deeper explanation of how each algorithm works and why, see [Understanding ESFS](docs/understanding_esfs.md).

### Helpful tips

- ESFS is designed to be used on raw counts matrices or read depth normalised counts matrices. We recommend that users do not apply additional data transformations or regress out potential confounders before running ESFS.
- The counts matrix you input to ESFS should have only basic quality control sample and gene filtering, such as removing cells with unusually low/high counts and genes expressed in only a small proportion of cells.
- Do not apply highly variable gene selection before running ESFS — one of the primary purposes of ESFS is to identify a set of informative genes from the larger gene set.
- The main parameter to change when identifying a set of genes informative of cell state is the `Num_Top_Ranked_Genes` parameter of the `plot_top_ranked_genes_UMAP()` function. We recommend starting at 3000 genes, plotting the cell UMAPs for the resulting gene clusters, and then trying values above or below 3000. You are looking for cell embeddings that improve data interpretability by revealing distinct cellular states with minimised batch effects. See supplemental figure S14 from the manuscript for a visualised example. 
- Having identified an optimised gene set and high-resolution embedding, proceed to applying ES-CCF and ES-FMG to identify robust marker genes that characterise distinct and hierarchical cellular populations in your data.


## Citation & Data

Please see our [manuscript](https://www.biorxiv.org/content/10.64898/2026.01.26.701684v1) for details regarding ESFS.

Datasets for reproducing the example workflows are available at the [figshare repository](https://figshare.com/s/4e445e7fa03cc4ccd289).

![ESFS is comprised of 3 main algorithms - ES-GSS, ES-CCF and ES-FMG](Figure_1.png)


> **Looking for the paper version?** Install v1.0.0 for exact reproducibility:
> ```
> pip install git+https://github.com/aradley/ESFS.git@v1.0.0
> ```
