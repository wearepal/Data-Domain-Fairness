"""Callbacks for images."""
from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from conduit.data import TernarySample
from conduit.types import Stage
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from ranzen import implements
from torch import Tensor
from torchvision.transforms import transforms

import wandb
from dfrdd.common import Denormalize

if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

__all__ = ["ImagesToLogger", "ImagesToLoggerDd"]


class ImageToLogger(pl.Callback):
    """Log Images."""

    def __init__(
        self,
        mean: list[float],
        std: list[float],
        log_every_n_epochs: int = 10,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 1,
    ) -> None:
        """Log validation images.

        Args:
            cfg: Data Config file for experiment.
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `torchvision` which is not installed yet."
            )

        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.denorm = Denormalize(mean=mean, std=std)

    @abstractmethod
    def log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Union[Tensor, Dict[str, Tensor]]],
        batch: TernarySample,
        batch_idx: int,
        stage: Stage,
    ) -> None:
        """Log Images to wandb."""

    @implements(pl.Callback)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Union[Tensor, Dict[str, Tensor]]],
        batch: TernarySample,
        batch_idx: int,
    ) -> None:
        if batch_idx % 100 == 0 or batch_idx == 1:
            self.log_images(trainer, pl_module, outputs, batch, batch_idx, Stage.fit)

    @implements(pl.Callback)
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Union[Tensor, Dict[str, Tensor]]],
        batch: TernarySample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.log_images(trainer, pl_module, outputs, batch, batch_idx, Stage.validate)

    @implements(pl.Callback)
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Union[Tensor, Dict[str, Tensor]]],
        batch: TernarySample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.log_images(trainer, pl_module, outputs, batch, batch_idx, Stage.test)

    def make_grid_and_log(
        self,
        caption: str,
        img: Tensor,
        pl_module: pl.LightningModule,
        stage: Stage,
        trainer: pl.Trainer,
    ) -> None:
        img = self.denorm(img)
        if len(img.size()) == 2:
            img_dim = pl_module.img_dim
            img = img.view(self.num_samples, *img_dim)

        grid = torchvision.utils.make_grid(
            tensor=img.clip(0, 1),
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{stage}/{pl_module.__class__.__name__}_{caption}_images"
        pl_module.logger.log_image(key=str_title, images=[transforms.ToPILImage()(grid)], caption=[caption])
        # trainer.logger.experiment.log(
        #     {str_title: wandb.Image(transforms.ToPILImage()(grid), caption=caption)},
        #     commit=False,
        # )


class ImagesToLogger(ImageToLogger):
    """Image logging callback."""

    def log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Union[Tensor, Dict[str, Tensor]]],
        batch: TernarySample,
        batch_idx: int,
        dataloader_idx: int,
        stage: Stage,
    ) -> None:
        """Callback that logs images."""
        if trainer.logger is not None and batch_idx == 1:
            image_batch = batch.x.to(pl_module.device)
            self.make_grid_and_log("original", image_batch, pl_module, stage, trainer)
            with torch.no_grad():
                pl_module.eval()
                _, model_out = pl_module(image_batch, batch.s)
            self.make_grid_and_log("predicted", model_out, pl_module, stage, trainer)

            for i in range(pl_module.card_s):
                _s = torch.ones_like(batch.s) * i

                with torch.no_grad():
                    pl_module.eval()
                    _, model_out = pl_module(image_batch, _s)

                self.make_grid_and_log(f"All_{i}", model_out, pl_module, stage, trainer)


class ImagesToLoggerDd(ImageToLogger):
    """Image logging callback."""

    def log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Union[Tensor, Dict[str, Tensor]]],
        batch: TernarySample,
        batch_idx: int,
        stage: Stage,
    ) -> None:
        """Callback that logs images."""
        if trainer.logger is not None and (
            (stage in (Stage.validate, Stage.test) and batch_idx == 1)
            or (stage == Stage.fit and batch_idx % 100 == 0)
        ):
            image_batch = batch.x.to(pl_module.device)
            self.make_grid_and_log("original", image_batch, pl_module, stage, trainer)
            if stage == Stage.fit:
                pl_module.freeze()
                _, debiased = pl_module(image_batch, batch.s)
                pl_module.unfreeze()
            else:
                _, debiased = pl_module(image_batch, batch.s)
            self.make_grid_and_log(
                "biased", batch.x - debiased, pl_module, stage, trainer
            )
            self.make_grid_and_log("debiased", debiased, pl_module, stage, trainer)
