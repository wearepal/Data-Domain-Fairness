from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import attr
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ranzen import implements
from ranzen.hydra import Option, Relay
from fairscale.nn import auto_wrap  # type: ignore
from dfrdd.conf import WandbLoggerConf

LOGGER = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=UserWarning)


@dataclass
class Config:
    """Base Config Schema."""

    _target_: str = "dfrdd.main.Config"
    data: Any = MISSING
    trainer: Any = MISSING
    model: Any = MISSING
    exp_group: Optional[str] = None
    seed: int = 888
    log_offline: bool = False


warnings.simplefilter(action="ignore", category=RuntimeWarning)


__all__ = ["DfddRelay"]


@attr.define(kw_only=True)
class DfddRelay(Relay):
    data: DictConfig
    model: DictConfig
    trainer: DictConfig
    logger: DictConfig
    checkpointer: DictConfig

    seed: Optional[int] = 42

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Path | str,
        *,
        data: list[Option],
        model: list[Option],
        clear_cache: bool = False,
    ) -> None:

        configs = dict(
            data=data,
            model=model,
            trainer=[Option(class_=pl.Trainer, name="base")],
            logger=[Option(class_=WandbLoggerConf, name="base")],
            checkpointer=[Option(class_=ModelCheckpoint, name="base")],
        )

        super().with_hydra(
            root=root,
            instantiate_recursively=False,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: Dict[str, Any]) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        pl.seed_everything(self.seed, workers=True)

        dm: pl.LightningDataModule = instantiate(self.data)
        dm.prepare_data()
        dm.setup()

        model: pl.LightningModule = instantiate(self.model)

        # enable parameter sharding with fairscale.
        # Note: when fully-sharded training is not enabled this is a no-op
        model = auto_wrap(model)  # type: ignore

        if self.logger.get("group", None) is None:
            default_group = (
                f"{dm.__class__.__name__.removesuffix('DataModule').lower()}_"
            )
            default_group += "_".join(
                dict_conf["_target_"].split(".")[-1].lower()
                for dict_conf in (self.data, self.model)
            )
            self.logger["group"] = default_group
        logger: WandbLogger = instantiate(self.logger, reinit=True)
        if raw_config is not None:
            logger.log_hyperparams(raw_config)  # type: ignore

        # Disable checkpointing when instantiating the trainer as we want to use
        # a hydra-instantiated checkpointer.
        trainer: pl.Trainer = instantiate(
            self.trainer,
            logger=logger,
            enable_checkpointing=False,
        )
        checkpointer: ModelCheckpoint = instantiate(self.checkpointer)
        trainer.callbacks.append(checkpointer)

        model.run(trainer=trainer, datamodule=dm, seed=self.seed)