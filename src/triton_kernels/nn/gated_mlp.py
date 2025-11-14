



def mlp_fwd(x, W1, W2, W3):
    return (torch.nn.functional.silu(x @ W2.T) * (x @ W1.T)) @ W3.T

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size:int = 64, intermediate_size:int = 256):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W3 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x:Tensor) -> Tensor:
        return mlp_fwd(x, self.W1.weight, self.W2.weight, self.W3.weight)