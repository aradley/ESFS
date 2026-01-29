"""
ESFS
"""

from . import backend as _backend_module
from .backend import use_cpu, use_gpu, use_mlx, configure, get_backend_info
# TODO: ESFS * import needs improved specificity to avoid namespace pollution
from .ESFS import *
from .plotting import *

# Re-assign backend module after star imports (ESFS.py exports 'backend' object)
backend = _backend_module
del _backend_module