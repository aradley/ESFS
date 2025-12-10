import warnings

import numpy as np
import scipy.sparse as spsparse

# Set default to CPU packages (numpy/scipy)

class ESFSBackend:
    xp = np
    xpsparse = spsparse
    using_gpu = False
    dtype = xp.float32

backend = ESFSBackend()

def _try_load_cupy():
    try:
        import cupy as _xp
        import cupyx.scipy.sparse as _xpsparse

        # Secondarily check if CUDA is available, in case it's run on e.g. different HPC partitions
        try:
            if _xp.cuda.is_available():
                return _xp, _xpsparse, True
            else:
                warnings.warn(
                    "CuPy is installed but CUDA is not available. ESFS will run on CPU, which may be slower for large datasets."
                )
                return np, spsparse, False
        # Needed because cupy.cuda.is_available() can raise exceptions
        # NOTE: This will be fixed in cupy v14
        # https://github.com/cupy/cupy/pull/9420
        except Exception:
            warnings.warn(
                "CuPy is installed but CUDA availability could not be determined. ESFS will run on CPU, which may be slower for large datasets."
            )
            return np, spsparse, False
    except ImportError:
        warnings.warn(
            "CuPy is not installed. ESFS will run on CPU, which may be slower for large datasets."
        )
        return np, spsparse, False


def use_gpu():
    """Force GPU backend (if available)."""
    xp, xpsparse, using_gpu = _try_load_cupy()
    backend.xp = xp
    backend.xpsparse = xpsparse
    backend.using_gpu = using_gpu


def use_cpu():
    """Force CPU backend."""
    backend.xp = np
    backend.xpsparse = spsparse
    backend.using_gpu = False

def configure(gpu: bool, upcast: bool = False):
    """Configure ESFS backend.

    Parameters
    ----------
    use_gpu
        Whether to use GPU backend (if available).
    upcast
        Whether to use float64 precision.
    """
    if gpu:
        use_gpu()
    else:
        use_cpu()
    if upcast:
        backend.dtype = backend.xp.float64
    else:
        backend.dtype = backend.xp.float32

# Try to use GPU backend by default, triggering on import
# Use float32 as default to avoid memory issues
configure(gpu=True, upcast=False)
