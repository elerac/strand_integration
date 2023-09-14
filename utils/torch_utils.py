import torch


def auto_torch_device() -> torch.device:
    """Select PyTorch backend automatically

    If CUDA is available, use CUDA.
    If CUDA is not available, but Metal Performance Shaders is available, use MPS.
    Otherwise, use CPU.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)


class WeightedMean:
    def __init__(self, weights: torch.Tensor) -> None:
        self.weights = weights
        self.sum = torch.sum(weights)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.weights * input) / self.sum


def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + torch.exp(-k * x))
