
import torch
from torch import Tensor
import os

def validate_tensor_device(x: Tensor):
    if not x.is_cuda:
        message = "Tensor must be on CUDA or TRITON_INTERPRET must be set to '1'"
        assert os.environ.get("TRITON_INTERPRET", False) == '1', message
    
def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def get_device() -> torch.device:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)
