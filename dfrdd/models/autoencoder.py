from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Optional, Union

import numpy
import pytorch_lightning as pl
import torch
from conduit.data import IMAGENET_STATS, CdtDataModule, TernarySample
from conduit.types import LRScheduler, Stage
from pl_bolts.models.autoencoders import resnet18_decoder, resnet18_encoder
from ranzen import implements
from ranzen.torch import TrainingMode
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import MeanAbsoluteError

__all__ = [
    "Ae",
    "AeGrl",
    "AeGrlEns",
    "AeGrlDo",
    "AeGrlMmd",
    "Mmd",
    "Gpd",
    "Vae",
    "VaeGrl",
    "VaeMmd",
    "Id",
]

from dfrdd.common import (
    ADV_LOSS,
    KLD_LOSS,
    MAE,
    MMD_LOSS,
    REC_LOSS,
    TO_MIN,
    Denormalize,
    FairnessType,
    Normalize,
)
from dfrdd.components.grad_proj import compute_grad, compute_proj_grads
from dfrdd.components.grad_reverse import grad_reverse
from dfrdd.components.hsic import hsic, kernel_matrix
from dfrdd.components.mmd import KernelType, mmd2


class AdvBase(nn.Module):
    """AE Adversary head."""

    def __init__(
        self,
        latent_dim: int,
        out_size: int,
        weight: float = 1.0,
    ):
        super().__init__()
        self.adv = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, out_size)),
            nn.Softmax(dim=-1),
        )
        self.weight = weight


class AdversaryGrl(AdvBase):
    @implements(nn.Module)
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        z_rev = grad_reverse(input_, lambda_=self.weight)
        return self.adv(z_rev)


class Adversary(AdvBase):
    @implements(nn.Module)
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.adv(input_)


class BaseAE(pl.LightningModule):
    def __init__(
        self,
        first_conv: bool = False,
        maxpool1: bool = False,
        latent_dim: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        fairness: Union[FairnessType, str] = FairnessType.DP,
    ):
        """
        Args:
            input_height: height of the images
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            latent_dim: dim of latent space
            lr: learning rate for AdamW
            weight_decay: weight decay for AdamW
        """
        super().__init__()
        self.save_hyperparameters()

        self.fairness = (
            FairnessType(fairness) if isinstance(fairness, str) else fairness
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq

        self.enc_out_dim = 512  # set according to the out_channel count of encoder used (512 for resnet18, 2048 for resnet50)
        self.latent_dim = latent_dim

        self.first_conv = first_conv
        self.max_pool1 = maxpool1

        self.encoder = nn.Sequential(
            resnet18_encoder(self.first_conv, self.max_pool1),
            nn.Linear(self.enc_out_dim, self.latent_dim),
        )

        self.maes = nn.ModuleDict({f"{stage}": MeanAbsoluteError() for stage in Stage})

        self.loss_fn = nn.MSELoss(reduction="sum")
        self.max_pixel_val = 255
        self.denormalizer = Denormalize(
            mean=IMAGENET_STATS.mean,
            std=IMAGENET_STATS.std,
            max_pixel_val=self.max_pixel_val,
        )
        self.normalizer = Normalize(
            mean=IMAGENET_STATS.mean,
            std=IMAGENET_STATS.std,
            max_pixel_val=self.max_pixel_val,
        )

    def _mae(
        self, stage: Stage, x_hat: torch.Tensor, target: torch.Tensor
    ) -> nn.Module:
        mae = self.maes[f"{stage}"]
        mae(x_hat.detach(), target)
        return mae

    def build(self, datamodule: CdtDataModule) -> None:
        self.decoder = resnet18_decoder(
            self.latent_dim + datamodule.card_s,
            datamodule.image_size,
            self.first_conv,
            self.max_pool1,
        )
        self.card_s = datamodule.card_s

    @abstractmethod
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        pass

    def _shared_step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> dict[str, torch.Tensor]:
        loss, logs, z = self.step(batch, batch_idx, stage)
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

    def _shared_epoch_end(self, outputs: dict[str, torch.Tensor], stage: Stage) -> None:
        mae = self.maes[f"{stage}"]
        self.log_dict({f"{stage}/{MAE}": mae})

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs: dict[str, torch.Tensor]) -> None:
        return self._shared_epoch_end(outputs, stage=Stage.validate)

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: dict[str, torch.Tensor]) -> None:
        return self._shared_epoch_end(outputs, stage=Stage.test)

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

    @implements(nn.Module)
    def forward(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(
            torch.cat([z, F.one_hot(s, num_classes=self.card_s)], dim=-1)
        )

    def fit(self, trainer: pl.Trainer, dm: CdtDataModule) -> None:
        trainer.fit(
            model=self,
            train_dataloaders=dm.train_dataloader(shuffle=True, drop_last=True),
            val_dataloaders=dm.val_dataloader(),
        )

    def test(
        self, trainer: pl.Trainer, dm: CdtDataModule, verbose: bool = True
    ) -> None:
        trainer.test(
            model=self,
            dataloaders=dm.test_dataloader(),
            verbose=verbose,
        )

    def run(
        self,
        *,
        datamodule: CdtDataModule,
        trainer: pl.Trainer,
        seed: Optional[int] = None,
    ) -> None:
        """Seed, build, fit, and test the model."""
        pl.seed_everything(seed)
        self.build(datamodule)
        self.fit(trainer=trainer, dm=datamodule)
        self.test(trainer=trainer, dm=datamodule)


class Id(BaseAE):
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        return (
            torch.ones_like(batch.s),
            {"loss": torch.zeros_like(batch.s).sum().sum()},
            batch.x,
        )

    def forward(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, self.denormalizer(x)


class Ae(BaseAE):
    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z = self.encoder(batch.x)
        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )

        recon_loss = self.loss_fn(x_hat, batch.x)
        mae = self._mae(stage, x_hat, batch.x)
        return (
            recon_loss,
            {f"{TO_MIN}": recon_loss, f"{REC_LOSS}": recon_loss, f"{MAE}": mae},
            z.detach(),
        )


class BaseAeAdv(BaseAE):
    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z = self.encoder(batch.x)
        adv = self.adversary(z)
        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )

        recon_loss = self.loss_fn(x_hat, batch.x)
        adv_loss = self.adv_loss(adv, batch.s)

        mae = self._mae(stage, x_hat, batch.x)
        total_loss = recon_loss + adv_loss
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                f"{ADV_LOSS}": adv_loss,
                f"{REC_LOSS}": recon_loss,
                f"{MAE}": mae,
            },
            z.detach(),
        )


class AeGrl(BaseAeAdv):
    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)
        self.adv_loss = nn.CrossEntropyLoss(reduction="mean")
        self.adversary = AdversaryGrl(self.latent_dim, datamodule.card_s, weight=1.0)


class AeGrlEns(BaseAeAdv):
    NUM_ADV = 5
    RESET_PROB = 0.05

    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)
        self.adv_loss = nn.CrossEntropyLoss(reduction="mean")
        self.adversary = nn.ModuleList(
            [
                AdversaryGrl(self.latent_dim, datamodule.card_s, weight=1.0)
                for _ in range(self.NUM_ADV)
            ]
        )

    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:

        for k, disc in enumerate(self.adversary):
            if numpy.random.uniform() < self.RESET_PROB:
                self.log("Reinitializing discriminator", k)
                self.adversary[k] = AdversaryGrl(
                    self.latent_dim, self.card_s, weight=1.0
                ).to(self.device)

        z = self.encoder(batch.x)
        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )
        recon_loss = self.loss_fn(x_hat, batch.x)

        adv_loss = torch.zeros_like(recon_loss)
        for adv in self.adversary:
            s_pred = adv(z)
            adv_loss += self.adv_loss(s_pred, batch.s)
        adv_loss /= len(self.adversary)

        total_loss = recon_loss + adv_loss
        mae = self._mae(stage, x_hat, batch.x)
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                f"{ADV_LOSS}": adv_loss,
                f"{REC_LOSS}": recon_loss,
                f"{MAE}": mae,
            },
            z.detach(),
        )


class AeGrlDo(BaseAeAdv):
    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)
        self.adv_loss = nn.CrossEntropyLoss(reduction="mean")
        self.adversary = nn.Sequential(
            nn.Dropout(0.5),
            AdversaryGrl(self.latent_dim, datamodule.card_s, weight=1.0),
        )


class Gpd(BaseAeAdv):
    """Zhang Mitigating Unwanted Biases."""

    def __init__(
        self,
        first_conv: bool = False,
        maxpool1: bool = False,
        latent_dim: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ):
        super().__init__(
            first_conv=first_conv,
            maxpool1=maxpool1,
            latent_dim=latent_dim,
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )
        self.automatic_optimization = False  # Mark for manual optimization
        self.adv_loss = nn.CrossEntropyLoss(reduction="mean")

    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)
        self.adv_loss = nn.CrossEntropyLoss(reduction="mean")
        self.adversary = Adversary(self.latent_dim, datamodule.card_s)

    def do_optimisation(
        self,
        stage: Stage,
        recon_loss: torch.Tensor,
        adv_loss: torch.Tensor,
        opt: torch.optim.Optimizer,
    ):
        if stage is Stage.fit:
            compute_proj_grads(
                model=self.encoder, loss_p=recon_loss, loss_a=adv_loss, alpha=1.0
            )
            compute_grad(model=self.adversary, loss=adv_loss)
            compute_grad(model=self.decoder, loss=recon_loss)
            opt.step()

            if (
                self.lr_sched_interval is TrainingMode.step
                or self.lr_sched_interval == "step"
            ) and self.global_step % self.lr_sched_freq == 0:
                sch = self.lr_schedulers()
                sch.step()
            if (
                self.lr_sched_interval is TrainingMode.epoch
                or self.lr_sched_interval == "epoch"
            ) and self.trainer.is_last_batch:
                sch = self.lr_schedulers()
                sch.step()

    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        opt = self.optimizers()
        opt.zero_grad()

        z = self.encoder(batch.x)
        s_pred = self.adversary(z)
        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )

        recon_loss = self.loss_fn(x_hat, batch.x)
        adv_loss = self.adv_loss(s_pred, batch.s)

        self.do_optimisation(stage, recon_loss, adv_loss, opt)

        mae = self._mae(stage, x_hat, batch.x)
        return (
            recon_loss,
            {
                f"{TO_MIN}": recon_loss,
                f"{REC_LOSS}": recon_loss,
                f"{ADV_LOSS}": adv_loss,
                f"{MAE}": mae,
            },
            z.detach(),
        )


class Mmd(BaseAE):
    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z = self.encoder(batch.x)
        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )

        recon_loss = self.loss_fn(x_hat, batch.x)
        if self.card_s == 2:
            mmd_loss = mmd2(z[batch.s == 0], z[batch.s == 1], kernel=KernelType.RBF)
        else:
            mmd_loss = torch.zeros_like(recon_loss)
            for sig in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
                kern_xx = kernel_matrix(x=z, sigma=sig)
                kern_ss = kernel_matrix(
                    x=F.one_hot(batch.s, num_classes=self.card_s), sigma=sig
                )
                mmd_loss += hsic(kern_x=kern_xx, kern_y=kern_ss, m=batch.x.shape[0])

        total_loss = recon_loss + mmd_loss

        mae = self._mae(stage, x_hat, batch.x)
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                f"{MMD_LOSS}": mmd_loss,
                f"{REC_LOSS}": recon_loss,
                f"{MAE}": mae,
            },
            z.detach(),
        )


class AeGrlMmd(BaseAeAdv):
    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)
        self.adv_loss = nn.CrossEntropyLoss(reduction="mean")
        self.adversary = AdversaryGrl(self.latent_dim, datamodule.card_s, weight=1.0)

    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z = self.encoder(batch.x)
        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )
        recon_loss = self.loss_fn(x_hat, batch.x)

        adv = self.adversary(z)
        adv_loss = self.adv_loss(adv, batch.s)

        if self.card_s == 2:
            mmd_loss = mmd2(z[batch.s == 0], z[batch.s == 1], kernel=KernelType.RBF)
        else:
            mmd_loss = torch.zeros_like(recon_loss)
            for sig in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
                kern_xx = kernel_matrix(x=z, sigma=sig)
                kern_ss = kernel_matrix(
                    x=F.one_hot(batch.s, num_classes=self.card_s), sigma=sig
                )
                mmd_loss += hsic(kern_x=kern_xx, kern_y=kern_ss, m=batch.x.shape[0])

        total_loss = recon_loss + mmd_loss + adv_loss

        mae = self._mae(stage, x_hat, batch.x)
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                f"{MMD_LOSS}": mmd_loss,
                f"{REC_LOSS}": recon_loss,
                f"{ADV_LOSS}": adv_loss,
                f"{MAE}": mae,
            },
            z.detach(),
        )


class Vae(BaseAE):
    def __init__(
        self,
        first_conv: bool = False,
        maxpool1: bool = False,
        latent_dim: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ):
        """
        Args:
            input_height: height of the images
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            latent_dim: dim of latent space
            lr: learning rate for AdamW
            weight_decay: weight decay for AdamW
        """
        super().__init__(
            first_conv=first_conv,
            maxpool1=maxpool1,
            latent_dim=latent_dim,
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )

        self.encoder = resnet18_encoder(self.first_conv, self.max_pool1)
        self.mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.loss_fn = nn.MSELoss(reduction="mean")

    def sample(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> tuple[
        torch.distributions.Distribution, torch.distributions.Distribution, torch.Tensor
    ]:
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)

    @implements(nn.Module)
    def forward(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(
            torch.cat([z, F.one_hot(s, num_classes=self.card_s)], dim=-1)
        )

    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        x = self.encoder(batch.x)
        mu = self.mu(x)
        log_var = self.var(x)
        p, q, z = self.sample(mu, log_var)
        if stage is Stage.fit:
            x_hat = self.decoder(
                torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
            )
        else:
            x_hat = self.decoder(
                torch.cat([mu, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
            )
        recon_loss = self.loss_fn(x_hat, batch.x)
        kl = torch.distributions.kl_divergence(q, p).mean()

        loss = kl + recon_loss

        mae = self._mae(stage, x_hat, batch.x)
        logs = {
            f"{TO_MIN}": loss,
            f"{REC_LOSS}": recon_loss,
            f"{KLD_LOSS}": kl,
            f"{MAE}": mae,
        }
        return loss, logs, z.detach()


class VaeGrl(Vae):
    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)
        self.adv_loss = nn.CrossEntropyLoss(reduction="mean")
        self.adversary = AdversaryGrl(self.latent_dim, datamodule.card_s)

    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        x = self.encoder(batch.x)
        mu = self.mu(x)
        log_var = self.var(x)
        p, q, z = self.sample(mu, log_var)

        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )

        recon_loss = self.loss_fn(x_hat, batch.x)
        kl = torch.distributions.kl_divergence(q, p).mean()

        s_pred = self.adversary(z)
        adv_loss = self.adv_loss(s_pred, batch.s)

        mae = self._mae(stage, x_hat, batch.x)
        total_loss = recon_loss + adv_loss + kl
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                f"{ADV_LOSS}": adv_loss,
                f"{REC_LOSS}": recon_loss,
                f"{KLD_LOSS}": kl,
                f"{MAE}": mae,
            },
            z.detach(),
        )


class VaeMmd(Vae):
    @implements(BaseAE)
    def build(self, datamodule: CdtDataModule) -> None:
        super().build(datamodule)

    @implements(BaseAE)
    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        x = self.encoder(batch.x)
        mu = self.mu(x)
        log_var = self.var(x)
        p, q, z = self.sample(mu, log_var)

        x_hat = self.decoder(
            torch.cat([z, F.one_hot(batch.s, num_classes=self.card_s)], dim=-1)
        )

        recon_loss = self.loss_fn(x_hat, batch.x)
        kl = torch.distributions.kl_divergence(q, p).mean()

        if self.card_s == 2:
            mmd_loss = mmd2(z[batch.s == 0], z[batch.s == 1], kernel=KernelType.RBF)
        else:
            mmd_loss = torch.zeros_like(recon_loss)
            for sig in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
                kern_xx = kernel_matrix(x=z, sigma=sig)
                kern_ss = kernel_matrix(
                    x=F.one_hot(batch.s, num_classes=self.card_s), sigma=sig
                )
                mmd_loss += hsic(kern_x=kern_xx, kern_y=kern_ss, m=batch.x.shape[0])

        mae = self._mae(stage, x_hat, batch.x)
        total_loss = recon_loss + mmd_loss + kl
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                f"{MMD_LOSS}": mmd_loss,
                f"{REC_LOSS}": recon_loss,
                f"{KLD_LOSS}": kl,
                f"{MAE}": mae,
            },
            z.detach(),
        )
