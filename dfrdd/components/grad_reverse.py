from __future__ import annotations

import torch
from torch import autograd


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx: autograd.Function, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        """Do GRL."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:  # type: ignore[override]
        """Do GRL."""
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Gradient Reversal layer."""
    return GradReverse.apply(features, lambda_)
