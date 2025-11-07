

from torch import nn, Tensor
import torch
from torch.autograd.function import Function
import triton
import triton.language as tl

from ..utils import validate_tensor_device

class FusedSoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        return softmax_triton(x)
    
    @staticmethod
    def backward(ctx, x):
        raise NotImplementedError

class FusedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x:Tensor) -> Tensor:
        # change_view = x.ndim > 1
        # if change_view:
        #     return FusedSoftmaxFunction.apply(x.view(-1)).view(x.shape)
        return FusedSoftmaxFunction.apply(x)
        


@triton.jit
def fused_softmax_kernel(x_pointer, y_pointer, x_stride, y_stride, n_rows, n_cols, block_size: tl.constexpr):
    # get the program id: each program of the grid handles one (or more) rows of the tensor
    pid = tl.program_id(axis=0)

    # strided execution: can run the program in a strided way (e.g. for row 0, 8, 16, ...)
    row_step = tl.num_programs(axis=0) # n. of programs running on given axis

    # loop through the rows executed by program with this pid
    for row_idx in tl.range(pid, n_rows, row_step):
        x_row_pointer = x_pointer + row_idx * x_stride

        col_offset = tl.arange(0, block_size)
        x_col_pointer = x_row_pointer + col_offset

        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = col_offset < n_cols

        # compute the softmax (with shift for numerical stab.)
        row = tl.load(x_col_pointer, mask, other=-float('inf'))

        row_minus_max = row - tl.max(row, axis=0)
        num = tl.exp(row_minus_max)
        den = tl.sum(num, axis=0)
        y = num / den

        y_row_pointer = y_pointer + row_idx * y_stride
        y_col_pointer = y_row_pointer + col_offset
        tl.store(y_col_pointer, y, mask)



def softmax_triton(x:Tensor, block_size:int=1024) -> Tensor:
    validate_tensor_device(x)

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    grid = n_rows,
    BLOCK_SIZE = triton.next_power_of_2(n_cols)  # Used to tile the row

    fused_softmax_kernel[grid](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE)

    return y

