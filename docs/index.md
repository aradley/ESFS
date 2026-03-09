# ESFS — Entropy Sorting Feature Selection

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/aradley/ESFS)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

ESFS is a feature selection package built on the **Entropy Sorting (ES)** mathematical framework. It is primarily designed for feature selection and marker gene identification in single-cell RNA sequencing datasets, and integrates directly with [AnnData](https://anndata.readthedocs.io/) and [Scanpy](https://scanpy.readthedocs.io/).

## Overview

ESFS provides three main algorithms:

![ESFS is comprised of 3 main algorithms — ES-GSS, ES-CCF, and ES-FMG](../Figure_1.png)

| Algorithm | Full name | Purpose |
|-----------|-----------|---------|
| **ES-GSS** | Entropy Sorting Gene Set Selection | Pairwise ES metric calculation between all features |
| **ES-CCF** | Entropy Sorting Combinatorial Cluster Finder | Find the combination of clusters that maximises ESS correlation for each gene |
| **ES-FMG** | Entropy Sorting Find Marker Genes | Select N genes that capture maximally distinct expression patterns |

## Quick install

```bash
pip install git+https://github.com/aradley/ESFS.git@memory_optimised
```

See [Installation](installation.md) for GPU acceleration options (NVIDIA/CUDA and Apple Silicon/MLX).

## Understanding ESFS

New to ESFS? The [Understanding ESFS](understanding_esfs.md) page explains the reasoning behind each algorithm — why gene expression space matters, how ES-GSS differs from HVG selection, why intentional over-clustering works, and how to read ES-FMG results.

## Getting started

The best way to get started is to browse the [example workflows](getting_started.md) included in the repository. The [parameter guide](getting_started.md#parameter-guide) explains how to tune the three key parameters for your data.

## API reference

Full documentation for all public functions is available in the [API Reference](api/index.md).

## Citation

Please cite the ESFS [manuscript](https://www.biorxiv.org/content/10.64898/2026.01.26.701684v1) if you use this package in your work. Datasets used in the example workflows are available at the [figshare repository](https://figshare.com/s/4e445e7fa03cc4ccd289).
