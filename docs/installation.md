# Installation

ESFS can be installed with CPU-only support or with optional GPU acceleration for NVIDIA (CUDA) or Apple Silicon (MLX) hardware. All three variants automatically fall back to CPU if no compatible GPU is detected, so any install will work on any machine.

## Requirements

- Python 3.10 or later
- A suitable environment manager (e.g. `uv`, `venv`, or `conda`) is strongly recommended

## Standard install (CPU)

The default install uses NumPy and Numba for all computations on CPU.

Install directly from GitHub:

```bash
pip install git+https://github.com/aradley/ESFS.git@memory_optimised
```

Or clone and install locally:

```bash
git clone git@github.com:aradley/ESFS.git
cd ESFS
git checkout memory_optimised
pip install .
```

> **Looking for the paper version (v1.0.0)?**
> ```bash
> pip install git+https://github.com/aradley/ESFS.git@v1.0.0
> ```

## GPU acceleration (NVIDIA/CUDA)

For large datasets on systems with NVIDIA GPUs, ESFS can accelerate the ES matrix calculations via [CuPy](https://cupy.dev/).

```bash
pip install "esfs[gpu] @ git+https://github.com/aradley/ESFS.git@memory_optimised"
```

Or clone and install:

```bash
git clone git@github.com:aradley/ESFS.git
cd ESFS
git checkout memory_optimised
pip install '.[gpu]'
```

### Important notes for CUDA users

Getting CuPy working correctly — especially on HPC clusters — can be tricky. Please read the following carefully before installing.

**Match the CuPy version to your CUDA toolkit**

CuPy ships separate packages for each CUDA version (e.g. `cupy-cuda12x` for CUDA 12.x). The default ESFS GPU install uses `cupy-cuda12x`. If your system has a different CUDA version, you will need to install the matching CuPy package manually:

```bash
# Check your CUDA version first
nvcc --version
# or
nvidia-smi

# Then install the matching CuPy package, for example for CUDA 11.x:
pip install cupy-cuda11x
```

**Using ESFS on an HPC cluster**

HPC systems typically require you to load a CUDA module before CuPy can access the GPU. If you install CuPy on a login node (where no GPU is present) or without loading the CUDA module, the installation may succeed but CuPy will fail at runtime.

Recommended workflow on HPC:

1. Start an interactive job on a GPU node (or submit a setup script), so the GPU and CUDA libraries are available.
2. Load the appropriate CUDA module before installing or running ESFS:
   ```bash
   module load cuda/12.x  # replace with your cluster's module name
   ```
3. Install ESFS with the GPU extras inside your job environment.
4. In your job scripts, always load the CUDA module before running Python.

**A common pitfall** is installing CuPy in one environment (e.g. against CUDA 12.2 on a login node) and then running on a compute node that has a different CUDA version loaded. This mismatch causes a runtime error even though the install appeared to succeed.

**Verify your CuPy installation**

After installing, you can confirm CuPy can see the GPU with:

```python
import cupy
print(cupy.cuda.runtime.runtimeGetVersion())  # e.g. 12020 for CUDA 12.2
```

If this raises an error, check that your CUDA module is loaded and that the CuPy version matches.

## GPU acceleration (Apple Silicon / MLX)

For Mac users with Apple Silicon (M1, M2, M3, etc.), ESFS supports GPU acceleration via Apple's [MLX](https://github.com/ml-explore/mlx) framework.

```bash
pip install "esfs[mlx] @ git+https://github.com/aradley/ESFS.git@memory_optimised"
```

Or clone and install:

```bash
git clone git@github.com:aradley/ESFS.git
cd ESFS
git checkout memory_optimised
pip install '.[mlx]'
```

### Important notes for MLX users

- Requires **macOS 13.5 or later** and an Apple Silicon chip (M1 or newer).
- MLX releases do not always keep pace with the latest macOS or Python versions. If the install fails, check the [MLX release notes](https://github.com/ml-explore/mlx/releases) for compatibility information before raising an issue.
- **float64 precision is not supported by MLX/Metal.** If you require float64, switch to the CPU backend:
  ```python
  esfs.use_cpu()
  esfs.configure(gpu=False, upcast=True)
  ```
- Note: GPU-accelerated KNN and UMAP (via cuML) are not available on Apple Silicon. Those steps fall back to CPU automatically.

## Backend auto-detection

After installation, ESFS automatically detects and selects the best available backend when you run `import esfs`. A banner is printed showing the active backend:

```
╔══════════════════════════════════════════════════════════════════════╗
║  ESFS - Entropy Sorting Feature Selection                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  ES matrix calculations: 🚀 CUDA/CuPy (NVIDIA GPU)                  ║
║  ES_FMG / ES_CCF:        💻 CPU (Numba JIT)                          ║
║  Precision: float32                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

Detection priority: **CUDA/CuPy → MLX/Metal → CPU (NumPy/Numba)**

You can switch backends manually at any time — see [Backend configuration](api/backend.md).
