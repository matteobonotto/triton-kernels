import os
from .utils import is_cuda_available

if not is_cuda_available():
    os.environ["TRITON_INTERPRET"] = "1"
