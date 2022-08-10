from typing import Mapping, Union

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
from dfrdd.models.autoencoder import BaseAE
from dfrdd.models.vgg import VGG, VggOut

IMAGE_FEATS_SIG = 1.0
SENS_FEATS_SIG = 0.5


class Frdd(BaseAE):
    def _build(self):
        self.pred_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.tv_loss = TotalVariation()
        self.fc_layer = nn.Linear(self.vgg.model.classifier[0].in_features, self.card_y)

    def build(self, datamodule: CdtDataModule) -> None:
        self.encoder = resnet18_encoder(self.first_conv, self.max_pool1)
        self.decoder = resnet18_decoder(
            self.enc_out_dim,
            datamodule.image_size,
            self.first_conv,
            self.max_pool1,
        )
        self.card_s = datamodule.card_s
        self.card_y = datamodule.card_y
        self.output_layers = {
            "block1_conv1": 1,
            "block2_conv1": 6,
            "block3_conv1": 11,
            "block4_conv1": 20,
            "block5_conv1": 29,
            "block5_conv2": 31,
        }
        self.vgg = VGG(self.output_layers)
        torch.autograd.set_detect_anomaly(True)
        self.denormalizer = Denormalize(
            mean=IMAGENET_STATS.mean,
            std=IMAGENET_STATS.std,
            max_pixel_val=self.max_pixel_val,
        )
        self.max_pixel_val = 1.0
        self._build()

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Mapping[str, Union[LRScheduler, int, TrainingMode]]:
        opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
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
        debiased_x_hat = self.decoder(z)
        return z, debiased_x_hat

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

        vgg: VggOut = self.vgg(batch.x)
        debiased_vgg: VggOut = self.vgg(debiased_x_hat)

        recon_loss = self.loss_fn(
            self.denormalizer(debiased_x_hat), self.denormalizer(batch.x)
        )
        y_hat = self.fc_layer(debiased_vgg.pool5)
        pred_loss = self.pred_loss_fn(y_hat, batch.y)

        biased_decomp_loss, debiased_decomp_loss = self.decomposition_loss(
            batch, vgg, debiased_vgg, recon_loss
        )

        tv_loss = self.tv_loss(debiased_x_hat).mean() * 1e-8

        mae = self._mae(
            stage, self.denormalizer(debiased_x_hat), self.denormalizer(batch.x)
        )
        total_loss = (
            biased_decomp_loss + debiased_decomp_loss + recon_loss + pred_loss + tv_loss
        )
        return (
            total_loss,
            {
                f"{TO_MIN}": total_loss,
                f"{MMD_LOSS}_biased": biased_decomp_loss,
                f"{MMD_LOSS}_debiased": debiased_decomp_loss,
                f"{REC_LOSS}": recon_loss,
                f"{MAE}": mae,
                f"{PRED_LOSS}": pred_loss,
                f"{TV_LOSS}": tv_loss,
            },
            z.detach(),
        )
