import pytest
import torch
from torch import Tensor, autograd, nn
import triton

from triton_kernels.utils import get_device, copy_weights
from triton_kernels.nn.linear import Linear

DEVICE = get_device()

test_tensors = [
    torch.rand(10),
    torch.rand(10, 20),
    torch.rand(10, 20, 30),
    torch.rand(10, 20, 30, 40),
]


@pytest.mark.parametrize("x", test_tensors)
def test_linear(x: Tensor):
    x = x.to(DEVICE)
    x.requires_grad = True
    
    linear = Linear(x.shape[-1],x.shape[-1], bias=True)
    linear_ref = nn.Linear(x.shape[-1], x.shape[-1], bias=True)
    copy_weights(linear_ref, linear,)
    
    out = linear(x)
    out_ref = linear_ref(x)
    triton.testing.assert_close(out, out_ref, atol=1e-6)
    
    # loss_ref = out_ref.sum()
    # loss_ref.backward()
    # print(linear_ref.weight.grad[0,0])
    
    # loss = out.sum()
    # loss.backward()
    # print(linear.weight.grad[0,0])
    
    
    grad_outputs = torch.rand_like(x).to(DEVICE)
    grads_ref = autograd.grad(out_ref, (x, linear_ref.weight, linear_ref.bias), grad_outputs=grad_outputs, retain_graph=True)
    grads = autograd.grad(out, (x, linear.weight, linear.bias), grad_outputs=grad_outputs, retain_graph=True)
    for g_r, g in zip(grads_ref, grads):
        triton.testing.assert_close(g_r, g, atol=1e-6)
    print("Done!!")
    
    
test_linear(test_tensors[-1])