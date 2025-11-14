
import torch
import pytest

from triton_kernels.functional.matmul import matmul


def test_matmul():
    M, N, K = (10, 11, 12)
    A = torch.rand(M.N)
    B = torch.rand(N,K)
    
    out_ref = A @ B
    out = matmul(A,B)
    