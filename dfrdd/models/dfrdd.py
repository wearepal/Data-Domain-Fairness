from typing import Mapping, Union, Optional

import torch
import torch.nn.functional as F
from conduit.data import IMAGENET_STATS, CdtDataModule, TernarySample
from conduit.types import LRScheduler, Stage
from kornia.losses import TotalVariation
from pl_bolts.models.autoencoders import resnet18_decoder, resnet18_encoder
from ranzen import implements
from ranzen.torch import TrainingMode
from torch import nn

__all__ = ["Frdd"]

import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import MeanAbsoluteError

from dfrdd.common import (
    MAE,
    MMD_LOSS,
    PRED_LOSS,
    REC_LOSS,
    SIG_VALUES,
    TO_MIN,
    TV_LOSS,
    Denormalize,
    FairnessType,
)
from dfrdd.components.hsic import hsic, kernel_matrix
from dfrdd.models.vgg import VGG, VggOut

IMAGE_FEATS_SIG = 1.0
SENS_FEATS_SIG = 0.5


class Frdd(pl.LightningModule):
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
        image_size: int = 64,
        card_s: int = 2,
        card_y: int = 2,
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
        self.image_size = image_size

        self.enc_out_dim = 512  # set according to the out_channel count of encoder used (512 for resnet18, 2048 for resnet50)
        self.latent_dim = latent_dim

        self.first_conv = first_conv
        self.max_pool1 = maxpool1

        self.encoder = nn.Sequential(
            resnet18_encoder(self.first_conv, self.max_pool1),
            nn.Linear(self.enc_out_dim, self.latent_dim),
        )

        self.maes = nn.ModuleDict({f"{stage}": MeanAbsoluteError() for stage in Stage})

        self.loss_fn = nn.MSELoss(reduction="mean")
        self.max_pixel_val = 255
        self.denormalizer = Denormalize(
            mean=IMAGENET_STATS.mean,
            std=IMAGENET_STATS.std,
            max_pixel_val=self.max_pixel_val,
        )

        self.output_layers = {
            "block3_conv1": 11,
            "block4_conv1": 20,
            "block5_conv1": 29,
        }
        self.vgg = VGG(self.output_layers)
        self.vgg.requires_grad_(False)
        self.encoder = nn.Sequential(
            resnet18_encoder(self.first_conv, self.max_pool1),
            nn.Linear(self.enc_out_dim, self.latent_dim),
        )
        self.encoder.requires_grad_(True)

        self.pred_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.tv_loss = TotalVariation()

        self.decoder = resnet18_decoder(
            self.latent_dim,
            self.image_size,
            self.first_conv,
            self.max_pool1,
        )
        self.decoder.requires_grad_(True)

        self.card_s = card_s
        self.card_y = card_y

        self.fc_layer = nn.Linear(self.vgg.model.classifier[0].in_features, self.card_y)

    def decomposition_loss(
        self,
        batch: TernarySample,
        vgg: VggOut,
        debiased_vgg: VggOut,
        example_loss: torch.Tensor,
    ):
        mask = 1 if self.fairness == FairnessType.EqOp else batch.y
        biased_blocks = (
            vgg.block3_conv1[batch.y == mask]
            - debiased_vgg.block3_conv1[batch.y == mask],
            vgg.block4_conv1[batch.y == mask]
            - debiased_vgg.block4_conv1[batch.y == mask],
            vgg.block5_conv1[batch.y == mask]
            - debiased_vgg.block5_conv1[batch.y == mask],
        )
        debiased_blocks = (
            debiased_vgg.block3_conv1[batch.y == mask],
            debiased_vgg.block4_conv1[batch.y == mask],
            debiased_vgg.block5_conv1[batch.y == mask],
        )

        biased_mmd_loss = torch.zeros_like(example_loss)
        for out in biased_blocks:
            kern_xx = kernel_matrix(x=out, sigma=IMAGE_FEATS_SIG)
            kern_ss = kernel_matrix(
                x=F.one_hot(batch.s[batch.y == mask], num_classes=self.card_s),
                sigma=SENS_FEATS_SIG,
            )
            biased_mmd_loss -= hsic(kern_x=kern_xx, kern_y=kern_ss, m=batch.x.shape[0])
        debiased_mmd_loss = torch.zeros_like(example_loss)
        for out in debiased_blocks:
            kern_xx = kernel_matrix(x=out, sigma=IMAGE_FEATS_SIG)
            kern_ss = kernel_matrix(
                x=F.one_hot(batch.s[batch.y == mask], num_classes=self.card_s),
                sigma=SENS_FEATS_SIG,
            )
            debiased_mmd_loss += hsic(
                kern_x=kern_xx, kern_y=kern_ss, m=batch.x.shape[0]
            )
        return biased_mmd_loss, debiased_mmd_loss

    def step(
        self, batch: TernarySample, batch_idx: int, stage: Stage
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z = self.encoder(batch.x)
        debiased_x_hat = self.decoder(z)

        # vgg: VggOut = self.vgg(batch.x)
        debiased_vgg: VggOut = self.vgg(debiased_x_hat)

        recon_loss = self.loss_fn(
            self.denormalizer(debiased_x_hat), self.denormalizer(batch.x)
        )
        y_hat = self.fc_layer(debiased_vgg.pool5)
        pred_loss = self.pred_loss_fn(y_hat, batch.y)

        # biased_decomp_loss, debiased_decomp_loss = self.decomposition_loss(
        #     batch, vgg, debiased_vgg, recon_loss
        # )

        # tv_loss = self.tv_loss(debiased_x_hat).mean() * 1e-8

        # mae = self._mae(
        #     stage, self.denormalizer(debiased_x_hat), self.denormalizer(batch.x)
        # )
        if self.current_epoch < 10:
            total_loss = recon_loss
        else:
            total_loss = (
                recon_loss
                + pred_loss  # + biased_decomp_loss + debiased_decomp_loss + tv_loss
            )
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                # f"{MMD_LOSS}_biased": biased_decomp_loss,
                # f"{MMD_LOSS}_debiased": debiased_decomp_loss,
                f"{REC_LOSS}": recon_loss,
                # f"{MAE}": mae,
                f"{PRED_LOSS}": pred_loss,
                # f"{TV_LOSS}": tv_loss,
            },
            z.detach(),
        )

    @implements(nn.Module)
    def forward(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        debiased_x_hat = self.decoder(z)
        return z, debiased_x_hat

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
        with torch.no_grad():
            return self._shared_step(batch, batch_idx, stage=Stage.validate)

    @implements(pl.LightningModule)
    def test_step(
        self, batch: TernarySample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
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
            filter(lambda p: p.requires_grad, self.parameters()),
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
        self.fit(trainer=trainer, dm=datamodule)
        self.test(trainer=trainer, dm=datamodule)
