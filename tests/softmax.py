
import torch
from torch import Tensor, nn
import triton

from triton_kernels.utils import get_device
from triton_kernels.nn import FusedSoftmax

DEVICE = get_device()


def test_fused_gelu():
    
    x = torch.rand(100, 100).to(DEVICE)
    triton.testing.assert_close(nn.Softmax()(x), FusedSoftmax()(x))


test_fused_gelu()