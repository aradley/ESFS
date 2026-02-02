# Changelog

All notable changes to ESFS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-02

### Initial Release

This is the first stable release of ESFS, accompanying our publication.

#### Features
- **ES-GSS** (Entropy Sorting Gene Set Scoring): Gene set scoring using entropy-based feature weighting
- **ES-CCF** (Entropy Sorting Cluster Characterising Features): Identification of features that characterise cell clusters
- **ES-FMG** (Entropy Sorting Feature Marker Genes): Marker gene identification using entropy sorting
- GPU acceleration via CuPy/CUDA for large datasets
- Integration with scanpy/anndata single-cell workflows
- Visualization utilities for ranked genes and UMAP projections

#### Dependencies
- Python >= 3.11
- Core: numpy, scipy, pandas, matplotlib, plotly
- Single-cell: anndata, scanpy
- Optional GPU: cupy-cuda12x

This version is archived for reproducibility. For the latest features, see newer releases.
