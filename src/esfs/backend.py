import warnings

import numpy as np
import scipy.sparse as spsparse

class ESFSBackend:
    xp = np
    xpsparse = spsparse
    using_gpu = False  # CuPy/CUDA GPU
    using_mlx = False  # MLX/Metal GPU (Apple Silicon)
    dtype = xp.float32

backend = ESFSBackend()

def _try_load_cupy():
    """Try to load CuPy/CUDA backend. Returns (xp, xpsparse, success)."""
    try:
        import cupy as _xp
        import cupyx.scipy.sparse as _xpsparse

        # Check if CUDA is available (may vary on HPC partitions)
        try:
            if _xp.cuda.is_available():
                return _xp, _xpsparse, True
            else:
                return np, spsparse, False
        # cupy.cuda.is_available() can raise exceptions (fixed in cupy v14)
        # https://github.com/cupy/cupy/pull/9420
        except Exception:
            return np, spsparse, False
    except ImportError:
        return np, spsparse, False


def use_gpu():
    """Force GPU backend (if available). Tries CuPy first, then MLX."""
    xp, xpsparse, using_gpu = _try_load_cupy()
    backend.xp = xp
    backend.xpsparse = xpsparse
    backend.using_gpu = using_gpu
    backend.using_mlx = False  # Reset MLX when using CuPy

    # If CuPy/CUDA not available, try MLX (Apple Silicon)
    if not using_gpu and _try_load_mlx():
        backend.using_mlx = True

    # Update module-level variables in ESFS and plotting
    _update_dependent_modules()


def use_cpu():
    """Force CPU backend."""
    backend.xp = np
    backend.xpsparse = spsparse
    backend.using_gpu = False
    backend.using_mlx = False

    # Update module-level variables in ESFS and plotting
    _update_dependent_modules()


def _try_load_mlx():
    """Try to load MLX backend for Apple Silicon GPUs. Returns True if available."""
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except ImportError:
        return False


def _update_dependent_modules():
    """Update module-level variables in ESFS and plotting modules."""
    try:
        from . import ESFS, plotting
        ESFS._update_module_backend()
        plotting._update_module_backend()
    except ImportError:
        # Modules not yet loaded during initial import
        pass


def use_mlx():
    """Force MLX/Metal backend (if available on Apple Silicon)."""
    if _try_load_mlx():
        backend.xp = np  # MLX uses numpy for non-kernel operations
        backend.xpsparse = spsparse
        backend.using_gpu = False
        backend.using_mlx = True
        # Update module-level variables in ESFS and plotting
        _update_dependent_modules()
    else:
        warnings.warn(
            "MLX is not available. ESFS will run on CPU."
        )
        use_cpu()

def configure(gpu: bool, upcast: bool = False, verbose: bool = True):
    """Configure ESFS backend.

    Parameters
    ----------
    gpu
        Whether to use GPU backend (if available).
    upcast
        Whether to use float64 precision. Default is False (float32) to save memory.
    verbose
        Whether to print backend information on configuration.
    """
    if gpu:
        use_gpu()
    else:
        use_cpu()
    if upcast:
        backend.dtype = backend.xp.float64
    else:
        backend.dtype = backend.xp.float32
    # Update module-level variables (use_gpu/use_cpu already call this,
    # but we need it here too in case only precision changed)
    _update_dependent_modules()

    if verbose:
        _print_backend_info()


def _print_backend_info():
    """Print current backend information and manual override instructions."""
    # Determine current backend
    if backend.using_gpu:
        backend_name = "CUDA/CuPy (NVIDIA GPU)"
        backend_icon = "🚀"
    elif backend.using_mlx:
        backend_name = "MLX/Metal (Apple Silicon GPU)"
        backend_icon = "🍎"
    else:
        backend_name = "CPU (NumPy/Numba)"
        backend_icon = "💻"

    # Determine precision
    precision = "float64" if backend.dtype == np.float64 else "float32"

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ESFS Backend: {backend_icon} {backend_name:<40}      ║
║  Precision: {precision:<10}                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  To manually switch backends, run one of:                            ║
║                                                                      ║
║    from esfs import backend                                          ║
║    backend.configure(gpu=False)           # Force CPU                ║
║    backend.configure(gpu=True)            # Auto-detect GPU          ║
║    backend.use_cpu()                      # Force CPU (NumPy/Numba)  ║
║    backend.use_mlx()                      # Force MLX (Apple Silicon)║
║    backend.use_gpu()                      # Force CUDA (NVIDIA)      ║
║                                                                      ║
║  For higher precision (uses more memory):                            ║
║    backend.configure(gpu=True, upcast=True)   # float64              ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def get_backend_info():
    """Return a string describing the current backend configuration."""
    if backend.using_gpu:
        backend_name = "CUDA/CuPy (NVIDIA GPU)"
    elif backend.using_mlx:
        backend_name = "MLX/Metal (Apple Silicon GPU)"
    else:
        backend_name = "CPU (NumPy/Numba)"
    precision = "float64" if backend.dtype == np.float64 else "float32"
    return f"Backend: {backend_name}, Precision: {precision}"


# Try to use GPU backend by default, triggering on import
# Use float32 as default for memory efficiency (can set upcast=True for float64)
configure(gpu=True, upcast=False, verbose=True)
