from triton_kernels.utils import get_device
from triton_kernels.nn import Softmax

import pytest
import os
import torch
from torch import Tensor, nn, autograd

import triton
import triton.language as tl
import math


DEVICE = get_device()

test_tensors = [
    torch.rand(100),
    torch.rand(10, 100),
    torch.rand(2, 3, 4),
]

@pytest.mark.parametrize('x', test_tensors)
def test_softmax_fwd(x: Tensor):
    x = x.to(DEVICE)
    triton.testing.assert_close(nn.functional.softmax(x), Softmax()(x))

# test_softmax_fwd(test_tensors[0])


@pytest.mark.parametrize('x', test_tensors)
def test_softmax(x: Tensor):
    x = x.to(DEVICE)
    x.requires_grad = True
    
    out = Softmax()(x)
    out_ref = nn.functional.softmax(x, dim=-1)
    triton.testing.assert_close(out, out_ref, atol=1e-6)
    
    grad_outputs = torch.rand_like(x).to(DEVICE)
    grads_ref = autograd.grad(out_ref, (x, ), grad_outputs=grad_outputs, retain_graph=True)
    grads = autograd.grad(out, (x, ), grad_outputs=grad_outputs, retain_graph=True)
    for g_r, g in zip(grads_ref, grads):
        triton.testing.assert_close(g_r, g, atol=1e-6)
    print("Done!!")
    
    
# test_softmax(test_tensors[0])