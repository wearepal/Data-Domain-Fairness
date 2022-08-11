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

