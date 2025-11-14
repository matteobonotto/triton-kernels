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
    
    # numbers of programs along M and M
    num_programs_m = tl.cdiv(M, block_size_M)
    num_programs_n = tl.cdiv(N, block_size_N)
    
    # Map the global pid to pid_m and pid_n
    # pid -> (pid_m, pid_n)
    pid_m = pid // num_programs_m
    pid_n = pid % num_programs_n
    
    # get the offeset of pointers for A and B depending on the related pid
    offset_m = pid_m + tl.arange(0, block_size_M)
    offset_n = pid_n + tl.arange(0, block_size_N)
    offset_k = tl.arange(0, block_size_K)
    
    a_tile_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_tile_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
    
    c = tl.zeros((block_size_M, block_size_N), dtype=tl.float32)
    for k in tl.range(0, K, block_size_K):
        mask = 
        tile_a = tl.load(a_tile_ptrs, mask_a, other=0)
        tile_b = tl.load(b_tile_ptrs, mask_b, other=0)
        
        c = tl.dot(tile_a, tile_b, c)
        
    c = c.to(tl.bfloat16)
    
    # get the pointers for C
    tl.store(tile_c_ptrs, c, mask)


def matmul(a: Tensor, b: Tensor):
    assert a.ndim == 2, "expected A to have ndim=2"
    assert b.ndim == 2, "expected B to have ndim=2"
    assert a.shape[1] == b.shape[0], "dimension mismatch"

    M, N = a.shape
    K = b.shape[1]

    c = torch.zeros(M, K).to(DEVICE)

    # these will be tuned via autotune and provided as metaparameters
    block_size_M, block_size_N, block_size_K = (64, 64, 64)
    group_size_M = 8

    
    grid = (ceil(M / block_size_M) * ceil(N / block_size_N), )

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
