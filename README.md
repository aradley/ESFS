# ESFS (Entropy Sorting Feature Selection)

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/aradley/ESFS)

**Version 2.0.0**

ESFS is an Entropy Sorting based feature selection package primarily developed for feature selection and marker gene identification in single cell RNA sequencing datasets.

## Documentation

Full documentation is available in the [`docs/`](docs/) folder:

- [Installation](docs/installation.md)
- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api/index.md)

## Software overview

Single-cell RNA sequencing (scRNA-seq) has transformed our ability to resolve cellular heterogeneity, but extracting meaningful signals remains challenging due to technical noise and batch effects. Most methods for denoising scRNA-seq data have focused on using latent representations such as principal component analysis and deep learning to prioritise biological signals. By contrast, despite its influence on downstream analyses, feature selection has received relatively limited attention, leading to widespread reliance on the comparatively simplistic strategy of highly variable gene selection. Here we present Entropy Sorting Feature Selection (ESFS), a modular, user-friendly framework that substantially improves the interpretability of scRNA-seq data. Notably, ESFS reveals complex expression dynamics that are obscured in latent representations. We demonstrate the utility of ESFS in diverse data: identifying coherent developmental programs across eight independent human embryo datasets without batch integration; resolving spatial gene expression in mouse colon missed by conventional analyses; disambiguating shared and tumour-specific microenvironments in glioblastoma; and disentangling spatial, temporal, and neurogenic programs in the developing mouse neural tube. Beyond delivering a powerful and user-friendly software that deepens insight into complex biological systems, our work establishes Entropy Sorting as a novel information theoretic for advanced data analysis methods.

### Helpful tips

- ESFS is designed to be used on raw counts matrices or read depth normalised counts matrices. We recommend that users do not apply additional data transformations or regress out potential confounders before running ESFS.
- The counts matrix you input to ESFS should have only basic quality control sample and gene filtering, such as removing cells with unusually low/high counts and genes expressed in only a small proportion of cells.
- Do not apply highly variable gene selection before running ESFS — one of the primary purposes of ESFS is to identify a set of informative genes from the larger gene set.
- The main parameter to change when identifying a set of genes informative of cell state is the `Num_Top_Ranked_Genes` parameter of the `plot_top_ranked_genes_UMAP()` function. We recommend starting at 3000 genes, plotting the cell UMAPs for the resulting gene clusters, and then trying values above or below 3000. You are looking for cell embeddings that improve data interpretability by revealing distinct cellular states with minimised batch effects.
- Having identified an optimised gene set and high-resolution embedding, proceed to applying ES-CCF and ES-FMG to identify robust marker genes that characterise distinct and hierarchical cellular populations in your data.


## Citation & Data

Please see our [manuscript](https://www.biorxiv.org/content/10.64898/2026.01.26.701684v1) for details regarding ESFS.

Datasets for reproducing the example workflows are available at the [figshare repository](https://figshare.com/s/4e445e7fa03cc4ccd289).

![ESFS is comprised of 3 main algorithms - ES-GSS, ES-CCF and ES-FMG](Figure_1.png)


> **Looking for the paper version?** Install v1.0.0 for exact reproducibility:
> ```
> pip install git+https://github.com/aradley/ESFS.git@v1.0.0
> ```
