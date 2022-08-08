import torch
from torch import nn


def compute_proj_grads(
    *, model: nn.Module, loss_p: torch.Tensor, loss_a: torch.Tensor, alpha: float
) -> None:
    """Computes the adversarial-gradient projection term.
    :param model: Model whose parameters the gradients are to be computed w.r.t.
    :param loss_p: Prediction loss.
    :param loss_a: Adversarial loss.
    :param alpha: Pre-factor for adversarial loss.
    """
    grad_p = torch.autograd.grad(loss_p, tuple(model.parameters()), retain_graph=True)
    grad_a = torch.autograd.grad(loss_a, tuple(model.parameters()), retain_graph=True)

    def _proj(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (
            b * torch.sum(a * b) / torch.sum(b * b).clamp(min=torch.finfo(b.dtype).eps)
        )

    grad_p = [p - _proj(p, a) - alpha * a for p, a in zip(grad_p, grad_a)]

    for param, grad in zip(model.parameters(), grad_p):
        param.grad = grad


def compute_grad(*, model: nn.Module, loss: torch.Tensor) -> None:
    """Computes the adversarial gradient projection term.
    :param model: Model whose parameters the gradients are to be computed w.r.t.
    :param loss: Adversarial loss.
    """
    grad_list = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)

    for param, grad in zip(model.parameters(), grad_list):
        param.grad = grad
