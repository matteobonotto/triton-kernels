import os
import torch
from torch import Tensor, nn

import triton
import triton.language as tl
import math

from triton_kernels.utils import get_device
from triton_kernels.kernels import FusedGelu

DEVICE = get_device()


def test_fused_gelu():
    x = torch.rand(100).to(DEVICE)
    triton.testing.assert_close(nn.GELU()(x), FusedGelu()(x))
    
    x = torch.rand(100, 100).to(DEVICE)
    triton.testing.assert_close(nn.GELU()(x), FusedGelu()(x))
