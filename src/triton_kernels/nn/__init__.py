from typing import Dict, List
from torch import nn

from .gelu import GELU
from .softmax import Softmax

KERNELS: List[nn.Module] = [
    GELU,
    Softmax,
]

__all__ = [x.__name__ for x in KERNELS]
