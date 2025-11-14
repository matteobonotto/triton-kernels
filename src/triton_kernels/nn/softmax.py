from torch import nn, Tensor
import torch
from torch.autograd.function import Function
import triton
import triton.language as tl

from ..utils import validate_tensor_device, validate_contiguous


@triton.jit
def _softmax_triton_fwd(
    x_pointer, y_pointer, x_stride, y_stride, n_rows, n_cols, block_size: tl.constexpr
):
    # get the program id: each program of the grid handles one (or more) rows of the tensor
    pid = tl.program_id(axis=0)

    # strided execution: can run the program in a strided way (e.g. for row 0, 8, 16, ...)
    row_step = tl.num_programs(axis=0)  # n. of programs running on given axis

    col_offset = tl.arange(0, block_size)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = col_offset < n_cols

    # loop through the rows executed by program with this pid
    for row_idx in tl.range(pid, n_rows, row_step):
        x_row_pointer = x_pointer + row_idx * x_stride
        x_col_pointer = x_row_pointer + col_offset

        # compute the softmax (with shift for numerical stab.)
        row = tl.load(x_col_pointer, mask, other=-float("inf"))

        row_minus_max = row - tl.max(row, axis=0)
        num = tl.exp(row_minus_max)
        den = tl.sum(num, axis=0)
        y = num / den

        y_row_pointer = y_pointer + row_idx * y_stride
        y_col_pointer = y_row_pointer + col_offset
        tl.store(y_col_pointer, y, mask)


def _softmax_fwd(x: Tensor, block_size: int = 1024) -> Tensor:
    validate_tensor_device(x)

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    grid = (n_rows,)
    block_size = triton.next_power_of_2(n_cols)  # Used to tile the row

    _softmax_triton_fwd[grid](
        x, y, x.stride(0), y.stride(0), n_rows, n_cols, block_size
    )
    return y


@triton.jit
def load_tensor_row(x_ptr, row_idx, x_stride, col_ptr_offset, mask, other):
    # row pointer
    row_ptr = x_ptr + row_idx * x_stride

    # col pointer
    col_ptr = row_ptr + col_ptr_offset

    # load all tensors
    return tl.load(col_ptr, mask, other=other)


@triton.jit
def _softmax_triton_bwd(
    s_ptr,
    grad_output_ptr,
    grad_input_ptr,
    s_stride,
    grad_output_stride,
    grad_input_stride,
    nrows,
    ncols,
    block_size: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)
    col_ptr_offset = tl.arange(0, block_size)
    mask = col_ptr_offset < ncols

    for row_idx in tl.range(pid, nrows, row_step):
        ### load s
        s = tl.load(s_ptr + row_idx * s_stride + col_ptr_offset, mask)

        ### load grad_output
        grad_output = tl.load(
            grad_output_ptr + row_idx * grad_output_stride + col_ptr_offset,
            mask,
        )

        # perform coputations
        # grad_input = s * grad_output
        alpha = tl.sum(s * grad_output, axis=0)
        grad_input = s * (grad_output - alpha)

        # store the results
        tl.store(
            grad_input_ptr + row_idx * grad_input_stride + col_ptr_offset,
            grad_input,
            mask,
        )


def _softmax_bwd(s: Tensor, grad_output: Tensor, block_size: int = 1024) -> Tensor:
    for x in (s, grad_output):
        validate_tensor_device(x)

    nrows, ncols = s.shape
    grad_input = torch.empty_like(grad_output).to(s.device)
    grid = (nrows,)

    block_size = triton.next_power_of_2(ncols)

    _softmax_triton_bwd[grid](
        s,
        grad_output,
        grad_input,
        s.stride(0),
        grad_output.stride(0),
        grad_input.stride(0),
        nrows,
        ncols,
        block_size,
    )

    return grad_input


class SoftmaxFunction(Function):
    def forward(ctx, x):
        s = _softmax_fwd(x)
        ctx.save_for_backward(s)
        return s

    def backward(ctx, grad_output):
        (s,) = ctx.saved_tensors
        # alpha = (s * grad_output).sum(dim=1, keepdim=True)
        # grad_input = s * (grad_output - alpha)
        grad_input = _softmax_bwd(s, grad_output)
        return grad_input


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            return SoftmaxFunction.apply(x[None, :]).squeeze()
        if x.ndim > 2:
            x = validate_contiguous(x)
            return SoftmaxFunction.apply(x.view(-1, x.shape[-1])).view(x.shape)
        return SoftmaxFunction.apply(x)


base_benchmark_kwargs = {
    "x_names": ["N"],  # argument names to use as an x-axis for the plot
    "x_vals": [
        128 * i for i in range(2, 100, 10)
    ],  # different possible values for `x_name`
    "line_arg": "provider",  # argument name whose value corresponds to a different line in the plot
    "line_vals": ["triton", "torch"],  # possible values for `line_arg``
    "line_names": ["Triton", "Torch"],  # label name for the lines
    "plot_name": "softmax",  # name for the plot. Used also as a file name for saving the plot.
    "args": {"M": 4096},  # values for function arguments not in `x_names` and `y_name`
}

_fwd = Softmax()


def fwd(x, provider):
    ### for benchmark olny!
    if provider == "torch":
        return torch.nn.functional.softmax(x, dim=-1)  # (x, axis=-1)
    elif provider == "triton":
        return _fwd(x)
    else:
        raise ValueError
