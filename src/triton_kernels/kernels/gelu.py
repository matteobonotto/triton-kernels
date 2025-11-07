
from torch import nn, Tensor
import torch
from torch.autograd.function import Function
import triton
import triton.language as tl
from math import ceil, pi

from ..utils import validate_tensor_device

@triton.jit
def gelu_kernel(x_pointer, out_pointer, num_elements, block_size: tl.constexpr):

    pid = tl.program_id(axis=0)
    pointer_offset = pid*block_size + tl.arange(0, block_size)

    mask = pointer_offset < num_elements

    x = tl.load(x_pointer + pointer_offset, mask)

    const = tl.sqrt(2 / pi)
    angle = const * (x + 0.044715 * x * x * x)
    tanh = (tl.exp(2 * angle) - 1) / (tl.exp(2 * angle) + 1)
    chunk_res = 0.5 * x * (1 + tanh)

    tl.store(out_pointer + pointer_offset, chunk_res, mask)


def gelu_triton(x:Tensor, block_size: int = 2048) -> Tensor:
    validate_tensor_device(x)
    
    num_elements = x.numel()
    grid = ceil(num_elements/block_size),
    out = torch.empty_like(x).to(x.device)

    gelu_kernel[grid](x, out, num_elements, block_size)
    return out


class FusedGeluFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        return gelu_triton(x)
    
    @staticmethod
    def backward(ctx, x):
        raise NotImplementedError

class FusedGelu(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x:Tensor) -> Tensor:
        change_view = x.ndim > 1
        if change_view:
            return FusedGeluFunction.apply(x.view(-1)).view(x.shape)
        return FusedGeluFunction.apply(x)
        
