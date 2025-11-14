from torch.autograd.function import Function


class LinearSoftmaxFunction(Function):
    @staticmethod
    def forward(
        ctx,
    ): ...

    @staticmethod
    def backward(
        ctx,
    ): ...


def linear_softmax(x):
    return LinearSoftmaxFunction.apply()
