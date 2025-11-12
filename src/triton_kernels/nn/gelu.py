from torch import nn, Tensor
import torch
from torch.autograd.function import Function
import triton
import triton.language as tl
from math import ceil, pi

from ..utils import validate_tensor_device, validate_contiguous
import math


@triton.jit
def _gelu_fwd_triton(
    x_ptr, tanh_ptr, phi_ptr, act_ptr, num_elements, block_size: tl.constexpr
):

    pid = tl.program_id(axis=0)
    ptr_offset = pid * block_size + tl.arange(0, block_size)

    mask = ptr_offset < num_elements

    x = tl.load(x_ptr + ptr_offset, mask)

    angle = tl.sqrt(2 / pi) * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * angle)
    tanh = (exp - 1) / (exp + 1)
    phi = 0.5 * (1 + tanh)
    chunk_act = x * phi

    tl.store(tanh_ptr + ptr_offset, tanh, mask)
    tl.store(phi_ptr + ptr_offset, phi, mask)
    tl.store(act_ptr + ptr_offset, chunk_act, mask)


def _gelu_fwd(x: Tensor, block_size: int = 2048) -> Tensor:
    validate_tensor_device(x)

    num_elements = x.numel()
    grid = (ceil(num_elements / block_size),)
    act = torch.empty_like(x).to(x.device)
    phi = torch.empty_like(x).to(x.device)
    tanh = torch.empty_like(x).to(x.device)

    _gelu_fwd_triton[grid](x, tanh, phi, act, num_elements, block_size)
    return act, tanh, phi


@triton.jit
def _gelu_bwd_triton(
    x_ptr, tanh_ptr, phi_ptr, derivative_ptr, num_elements, block_size: tl.constexpr
):

    pid = tl.program_id(axis=0)
    ptr_offset = pid * block_size + tl.arange(0, block_size)
    mask = ptr_offset < num_elements

    x = tl.load(x_ptr + ptr_offset, mask)
    tanh = tl.load(tanh_ptr + ptr_offset, mask)
    phi = tl.load(phi_ptr + ptr_offset, mask)

    factor = 1 / tl.sqrt(2 * pi)
    # angle = tl.sqrt(2 / pi) * (x + 0.044715 * x ** 3)
    # tanh = nn.functional.tanh(gx)
    phi_prime = factor * (1 - tanh * tanh) * (1 + 3 * 0.044715 * x * x)

    # phi = 0.5 * (1 + tanh)

    derivative = x * phi_prime + phi

    tl.store(derivative_ptr + ptr_offset, derivative, mask)


def _gelu_bwd(x: Tensor, tanh: Tensor, phi: Tensor, block_size: int = 2048) -> Tensor:
    for t in [x, tanh, phi]:
        validate_tensor_device(t)

    num_elements = x.numel()
    grid = (ceil(num_elements / block_size),)

    derivative = torch.empty_like(x).to(x.device)

    _gelu_bwd_triton[grid](x, tanh, phi, derivative, num_elements, block_size)

    return derivative


class GeluFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        act, tanh, phi = _gelu_fwd(x)
        ctx.save_for_backward(x, tanh, phi)
        # return _gelu(x)
        return act

    @staticmethod
    def backward(ctx, grad_output):
        x, tanh, phi = ctx.saved_tensors

        # factor = 1 / (2 * math.pi) ** 0.5
        # gx = (2 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3)
        # tanh = nn.functional.tanh(gx)
        # phi_prime = factor * (1 - tanh**2) * (1 + 3 * 0.044715 * x ** 2)
        # phi = 0.5 * (1 + tanh)

        # derivative = phi + x * phi_prime
        derivative = _gelu_bwd(x, tanh, phi)
        return derivative * grad_output


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim > 1:
            x = validate_contiguous(x)
            return GeluFunction.apply(x.view(-1)).view(x.shape)
        return GeluFunction.apply(x)




base_benchmark_kwargs = {
    "x_names":['N'],  # argument names to use as an x-axis for the plot
    "x_vals":[128 * i for i in range(2, 100, 10)],  # different possible values for `x_name`
    "line_arg":'provider',  # argument name whose value corresponds to a different line in the plot
    "line_vals":['triton', 'torch'],  # possible values for `line_arg``
    "line_names":["Triton", "Torch"],  # label name for the lines
    "plot_name":"gelu",  # name for the plot. Used also as a file name for saving the plot.
    "args":{'M': 4096} # values for function arguments not in `x_names` and `y_name`
}

_fwd = GELU()

def fwd(x, provider):
    ### for benchmark olny! 
    if provider == "torch":
        return torch.nn.functional.gelu(x)
    elif provider == 'triton':
        return _fwd(x)
    else:
        raise ValueError
