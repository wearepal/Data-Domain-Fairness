from enum import Enum

import torch
from torch import nn

SIG_VALUES = (0.5,)  # , 1.0, 2.0, 5.0, 10.0, 20.0)

TO_MIN = "loss"
REC_LOSS = "rec_loss"
ADV_LOSS = "adv_loss"
MMD_LOSS = "mmd_loss"
KLD_LOSS = "kld_loss"
PRED_LOSS = "pred_loss"
TV_LOSS = "tv_loss"
MAE = "MAE"
S_ACC = "S-Acc"
Y_ACC = "Y-Acc"


class Target(Enum):
    S = "S"
    Y = "Y"


class FairnessType(Enum):
    DP = "DP"
    EO = "EO"
    EqOp = "EqOp"
    No = "No"


class Denormalize(nn.Module):
    def __init__(self, mean: list[float], std: list[float], max_pixel_val: int = 255):
        super().__init__()
        self.mean = mean
        self.std = std
        self.max_px_val = max_pixel_val

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`
        img = img * torch.tensor(
            [a * self.max_px_val for a in self.std],
            device=img.device,
            requires_grad=False,
        ).view(1, -1, 1, 1) + torch.tensor(
            [a * self.max_px_val for a in self.mean],
            device=img.device,
            requires_grad=False,
        ).view(
            1, -1, 1, 1
        )
        return img


class Normalize(nn.Module):
    def __init__(self, mean: list[float], std: list[float], max_pixel_val: int = 255):
        super().__init__()
        self.mean = mean
        self.std = std
        self.max_px_val = max_pixel_val

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = (
            img
            - torch.tensor(
                [a * self.max_px_val for a in self.mean],
                device=img.device,
                requires_grad=False,
            ).view(1, -1, 1, 1)
        ) / (
            torch.tensor(
                [a * self.max_px_val for a in self.std],
                device=img.device,
                requires_grad=False,
            ).view(1, -1, 1, 1)
        )
        return img
