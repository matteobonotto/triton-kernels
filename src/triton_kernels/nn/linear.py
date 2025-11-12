
from typing import Optional
from torch import Tensor, nn
from torch.autograd.function import Function
import triton

from ..utils import validate_contiguous, validate_tensor_device

@triton.jit()
def _linear_fwd_triton():
    ...
    
    
def _linear_fwd(x:Tensor, W:Tensor, b:Optional[Tensor] = None) -> Tensor:
    if b:
        return x @ W.T + b
    return x @ W.T

@triton.jit()
def _linear_bwd_triton():
    ...
    
    
def _linear_bwd(x, W, b, grad_output):
    grad_x = grad_output @ W
    grad_W = grad_output.T @ x
    grad_b = None #grad_output.sum(dim=0)
    return grad_x, grad_W, grad_b
    

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, W:Tensor, b:Tensor):
        ctx.save_for_backward(x)
        return _linear_fwd(x, W, b)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor, x: Tensor, W: Tensor, b: Optional[Tensor]):
        x, = ctx.saved_tensors
        grad_x, grad_W, grad_b = _linear_bwd(x, W, b, grad_output)
        return grad_x, grad_W, grad_b
    
    
class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x:Tensor) -> Tensor:
        if x.ndim > 2:
            x = validate_contiguous(x)
            new_shape = (1, ) + x.shape[-2:]
            return LinearFunction.apply(
                x.view(new_shape), 
                self.weight.data, 
                self.bias.data if self.bias else None,
            ).view(x.shape)
        return LinearFunction.apply(
                x, 
                self.weight.data, 
                self.bias.data if self.bias else None,
            )




base_benchmark_kwargs = {
    "x_names":['N'],  # argument names to use as an x-axis for the plot
    "x_vals":[128 * i for i in range(2, 100, 10)],  # different possible values for `x_name`
    "line_arg":'provider',  # argument name whose value corresponds to a different line in the plot
    "line_vals":['triton', 'torch'],  # possible values for `line_arg``
    "line_names":["Triton", "Torch"],  # label name for the lines
    "plot_name":"linear",  # name for the plot. Used also as a file name for saving the plot.
    "args":{'M': 4096} # values for function arguments not in `x_names` and `y_name`
}

M = base_benchmark_kwargs["args"]['M']
_fwd = Linear(M, M, bias=False)
_fwd_ref = nn.Linear(M, M, bias=False)

def fwd(x, provider):
    ### for benchmark olny! 
    if provider == "torch":
        return _fwd_ref(x)
    elif provider == 'triton':
        return _fwd(x)
    else:
        raise ValueError












