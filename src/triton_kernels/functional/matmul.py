from torch import nn, Tensor
import torch
import triton
import triton.language as tl
from math import ceil
from ..utils import get_device

DEVICE = get_device()


@triton.jit()
def _matmul_triton(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cn,
        stride_ck,
        block_size_M: tl.constexpr,
        block_size_N: tl.constexpr,
        block_size_K: tl.constexpr,
        group_size_M: tl.constexpr,
    ): 
    
    pid = tl.program_id(axis=0)


def matmul(a: Tensor, b: Tensor):
    assert a.ndim == 2, "expected A to have ndim=2"
    assert b.ndim == 2, "expected B to have ndim=2"
    assert a.shape[1] == b.shape[0], "dimension mismatch"

    M, N = a.shape
    K = b.shape[1]

    c = torch.zeros(M, K).to(DEVICE)

    # these will be tuned via autotune
    block_size_M, block_size_N, block_size_K = (64, 64, 64)
    group_size_M = 8

    grid = ceil(M / block_size_M), ceil(N / block_size_N)

    _matmul_triton[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_size_M,
        block_size_N,
        block_size_K,
        group_size_M,
    )

    return c
