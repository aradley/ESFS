"""
ESFS - Entropy Sorting Feature Selection
"""

__version__ = "1.0.0"

from .backend import use_cpu, use_gpu, configure
# TODO: ESFS * import needs improved specificity to avoid namespace pollution
from .ESFS import *
from .plotting import *