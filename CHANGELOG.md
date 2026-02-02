# Changelog

All notable changes to ESFS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

### Added
- Apple Silicon GPU support via MLX/Metal backend
- `use_mlx()` function for explicit MLX backend selection
- `get_backend_info()` function to query current backend status
- Flexible dependency version ranges for better compatibility
- Support for Python 3.11, 3.12, and 3.13
- Development dependencies (pytest, ruff)
- Numba integration for optimized CPU operations

### Changed
- Improved memory efficiency in correlation calculations
- Enhanced backend auto-detection logic
- Cleaner public API with explicit `__all__` exports
- Updated documentation for multi-backend support
- Relaxed dependency pinning for broader compatibility

### Fixed
- Handle all-NaN slices in nanargmax operations
- Various bug fixes in CUDA backend

## [1.0.0] - 2026-02-02

Initial paper release. See [v1.0.0 tag](https://github.com/aradley/ESFS/releases/tag/v1.0.0).

### Features
- ES-GSS (Entropy Sorting Gene Set Scoring)
- ES-CCF (Entropy Sorting Cluster Characterising Features)
- ES-FMG (Entropy Sorting Feature Marker Genes)
- GPU acceleration via CuPy/CUDA
- Integration with scanpy/anndata workflows
