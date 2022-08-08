from typing import Any, Mapping, Union

import pytorch_lightning as pl
import torch
from conduit.data import TernarySample
from conduit.types import LRScheduler, Stage
from ranzen import implements
from ranzen.torch import TrainingMode
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import Accuracy

from dfrdd.common import TO_MIN, Target


class Erm(pl.LightningModule):
    def __init__(
        self,
        recon_model: nn.Module,
        lr: float,
        weight_decay: float,
        lr_initial_restart: int,
        lr_restart_mult: float,
        lr_sched_interval: LRScheduler,
        lr_sched_freq: int,
        target: Target,
    ):
        super().__init__()
        self.recon_model = recon_model
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            # nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq
        self.accs = nn.ModuleDict({f"{stage}": Accuracy() for stage in Stage})
        self.loss = nn.CrossEntropyLoss()
        self.target = target

    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[Any, dict[str, Module]]:
        _, x_hat = self.recon_model(batch.x, batch.s)
        out = self.net(x_hat)
        loss = (
            self.loss(out, batch.y)
            if self.target == Target.Y
            else self.loss(out, batch.s)
        )
        acc = self.accs[f"{stage}"]
        acc(out.detach(), batch.y) if self.target == Target.Y else acc(
            out.detach(), batch.s
        )
        return loss, {f"{stage}/acc": acc}

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Mapping[str, Union[LRScheduler, int, TrainingMode]]:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": opt,
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer=opt, T_0=self.lr_initial_restart, T_mult=self.lr_restart_mult
            ),
            "interval": self.lr_sched_interval.name,
            "frequency": self.lr_sched_freq,
        }

    def _shared_step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> dict[str, torch.Tensor]:
        loss, logs = self.step(batch, batch_idx, stage)
        self.log_dict(
            {f"{stage}/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return {f"{TO_MIN}": loss}

    @implements(pl.LightningModule)
    def training_step(
        self, batch: TernarySample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self._shared_step(batch, batch_idx, stage=Stage.fit)

    @implements(pl.LightningModule)
    def validation_step(
        self, batch: TernarySample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self._shared_step(batch, batch_idx, stage=Stage.validate)

    @implements(pl.LightningModule)
    def test_step(
        self, batch: TernarySample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self._shared_step(batch, batch_idx, stage=Stage.test)
