import sys

import numpy
import pytorch_lightning as pl
import torch
from conduit.data import TernarySample
from conduit.types import Stage

__all__ = ["Znorm", "Zmean"]


class Znorm(pl.Callback):
    def calc_norm(
        self, z: torch.Tensor, pl_module: pl.LightningModule, stage: Stage
    ) -> None:
        z_norm = z.detach().norm(dim=1).mean()
        pl_module.log(f"{stage}/z_norm", z_norm)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: TernarySample,
        batch_idx: int,
    ) -> None:
        self.calc_norm(outputs["z"], pl_module, Stage.fit)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: TernarySample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.calc_norm(outputs["z"], pl_module, Stage.validate)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: TernarySample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.calc_norm(outputs["z"], pl_module, Stage.test)


class Zmean(pl.Callback):
    def calc_mean(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        pl_module: pl.LightningModule,
        stage: Stage,
    ) -> None:
        means = [z[s == _s].detach().mean() for _s in range(pl_module.card_s)]
        means = [0 if numpy.isnan(m.detach().cpu().numpy()) else m for m in means]

        minscan = sys.maxsize
        solution = max(x - (minscan := min(minscan, x)) for x in means)

        pl_module.log(f"{stage}/z_mean_max_diff", solution)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: TernarySample,
        batch_idx: int,
    ) -> None:
        self.calc_mean(outputs["z"], batch.s, pl_module, Stage.fit)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: TernarySample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.calc_mean(outputs["z"], batch.s, pl_module, Stage.validate)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: TernarySample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.calc_mean(outputs["z"], batch.s, pl_module, Stage.test)
