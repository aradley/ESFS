import warnings

import numpy as np
import scipy.sparse as spsparse

# Set default to CPU packages (numpy/scipy)
xp = np
xpsparse = spsparse
USING_GPU = False


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
    global xp, xpsparse, USING_GPU
    xp, xpsparse, USING_GPU = _try_load_cupy()


def use_cpu():
    """Force CPU backend."""
    global xp, xpsparse, USING_GPU
    xp, xpsparse, USING_GPU = np, spsparse, False


# Try to use GPU backend by default, triggering on import
use_gpu()
