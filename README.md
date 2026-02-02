# ESFS

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/aradley/ESFS)

**Version 2.0.0**

ESFS is an Entropy Sorting based feature selection package primarily developed for feature selection and marker gene identification in single cell RNA sequencing datasets.

> **Looking for the paper version?** Install v1.0.0 for exact reproducibility:
> ```
> pip install git+https://github.com/aradley/ESFS.git@v1.0.0
> ```

Please see our manuscript for details regarding ESFS -

Go to the Example_Workflows folder to see some example workflows that you may adapt for your own data.

Datasets for reproducing the example workflows may be found at the following Mendeley Data repository -

## Software overview

![ESFS is comprised of 3 main algorithms - ES-GSS, ES-CCF and ES-FMG](Figure_1.png)

### Installation

Install the latest version:

```
pip install git+https://github.com/aradley/ESFS.git
```

or clone and then install:

```
git clone git@github.com:aradley/ESFS.git
cd ESFS
pip install .
```

You should do this within an environment, using something like `uv`, `venv`, or `conda`.

## GPU acceleration (NVIDIA/CUDA)

For large datasets on systems with NVIDIA GPUs, ESFS supports GPU acceleration via CuPy:

```
pip install "esfs[gpu] @ git+https://github.com/aradley/ESFS.git@memory_optimised"
```

or clone and then install:

```
git clone -b memory_optimised git@github.com:aradley/ESFS.git
cd ESFS
pip install '.[gpu]'
```

This requires a compatible version of CUDA to be installed on your machine.

## Apple Silicon GPU acceleration (MLX)

For Mac users with Apple Silicon (M1/M2/M3, etc.), ESFS supports GPU acceleration via Apple's MLX framework:

```
pip install "esfs[mlx] @ git+https://github.com/aradley/ESFS.git@memory_optimised"
```

or clone and then install:

```
git clone -b memory_optimised git@github.com:aradley/ESFS.git
cd ESFS
pip install '.[mlx]'
```

## Backend configuration

By default, ESFS will auto-detect and use the best available backend when you run `import esfs`:
- **CUDA/CuPy** (NVIDIA GPU) - if CuPy is installed and CUDA is available
- **MLX/Metal** (Apple Silicon GPU) - if MLX is installed and Metal is available
- **CPU (NumPy/Numba)** - fallback if no GPU backend is available

A banner will display showing which backend is active.

### Switching backends manually

You can switch backends at runtime after importing ESFS:

```python
import esfs

# Force CPU (NumPy/Numba)
esfs.backend.use_cpu()

# Force CUDA (NVIDIA GPU)
esfs.backend.use_gpu()

# Force MLX (Apple Silicon GPU)
esfs.backend.use_mlx()

# Or use configure() for more control
esfs.backend.configure(gpu=False)           # Force CPU
esfs.backend.configure(gpu=True)            # Auto-detect GPU
esfs.backend.configure(gpu=True, upcast=True)  # GPU with float64 precision
```

### Precision

By default, ESFS uses float32 precision to save memory. For higher precision (at the cost of more memory), use:

```python
esfs.backend.configure(gpu=True, upcast=True)  # float64 precision
```

Note: MLX/Metal does not support float64. If you need float64 precision on Apple Silicon, use the CPU backend.
