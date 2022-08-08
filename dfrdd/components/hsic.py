import torch


def hsic(kern_x: torch.Tensor, *, kern_y: torch.Tensor, m: int) -> torch.Tensor:
    xy = torch.matmul(kern_x, kern_y)
    h = (
        torch.trace(xy) / m**2
        + torch.mean(kern_x) * torch.mean(kern_y)
        - 2 * torch.mean(xy) / m
    )
    return h * (m / (m - 1)) ** 2


def kernel_matrix(x: torch.Tensor, *, sigma: float) -> torch.Tensor:
    dim = len(x.size())
    x1 = torch.unsqueeze(x, 0)
    x2 = torch.unsqueeze(x, 1)
    if dim <= 1:
        return torch.exp(-0.5 * torch.pow(x1 - x2, 2) / sigma**2)
    axis = tuple(range(2, dim + 1))
    return torch.exp(-0.5 * torch.sum(torch.pow(x1 - x2, 2), dim=axis) / sigma**2)
