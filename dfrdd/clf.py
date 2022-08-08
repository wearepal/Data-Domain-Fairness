from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from types import MethodType
from typing import Any, Final, Mapping, Optional, TypeVar

import hydra
import pytorch_lightning as pl
import torch
from conduit.hydra.conduit.data.datamodules.conf import (
    CelebADataModuleConf,
    ColoredMNISTDataModuleConf,
    WaterbirdsDataModuleConf,
)
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from dfrdd.common import Target
from dfrdd.conf.classes.dfrdd.models.configs import (
    AeConf,
    AeGrlConf,
    AeGrlDoConf,
    AeGrlEnsConf,
    AeGrlMmdConf,
    DfrddConf,
    FrddConf,
    GpdConf,
    IdConf,
    MmdConf,
    VaeConf,
    VaeGrlConf,
    VaeMmdConf,
)
from dfrdd.conf.classes.pytorch_lightning.trainer.configs import TrainerConf
from dfrdd.models.erm import Erm

T = TypeVar("T")


LOGGER = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=UserWarning)


@dataclass
class ErmConfig:
    """Base Config Schema."""

    _target_: str = "dfrdd.clf.ErmConfig"
    data: Any = MISSING
    test_data: Any = MISSING
    trainer: Any = MISSING
    model: Any = MISSING
    exp_group: Optional[str] = None
    seed: int = 888
    log_offline: bool = False
    model_path: Optional[str] = None
    target: Optional[str] = "Y"


CS = ConfigStore.instance()
CS.store(name="config_schema", node=ErmConfig)  # General Schema
CS.store(name="trainer", node=TrainerConf, package="trainer", group="schema/trainer")

DATA_PKG: Final[str] = "data"  # package:dir_within_config_path
DATA_GROUP: Final[str] = "schema/data"  # group
CS.store(
    name="cmnist", node=ColoredMNISTDataModuleConf, package=DATA_PKG, group=DATA_GROUP
)
CS.store(name="celeba", node=CelebADataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(
    name="waterbirds", node=WaterbirdsDataModuleConf, package=DATA_PKG, group=DATA_GROUP
)

TEST_DATA_PKG: Final[str] = "test_data"  # package:dir_within_config_path
TEST_DATA_GROUP: Final[str] = "schema/test_data"  # group
CS.store(
    name="cmnist",
    node=ColoredMNISTDataModuleConf,
    package=TEST_DATA_PKG,
    group=TEST_DATA_GROUP,
)
CS.store(
    name="celeba",
    node=CelebADataModuleConf,
    package=TEST_DATA_PKG,
    group=TEST_DATA_GROUP,
)
CS.store(
    name="waterbirds",
    node=WaterbirdsDataModuleConf,
    package=TEST_DATA_PKG,
    group=TEST_DATA_GROUP,
)

CLF_PKG: Final[str] = "model"
CLF_GROUP: Final[str] = "schema/model"
CS.store(name="ae", node=AeConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="aegrl", node=AeGrlConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="aegrldo", node=AeGrlDoConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="aegrlens", node=AeGrlEnsConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="aegrlmmd", node=AeGrlMmdConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="mmd", node=MmdConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="gpd", node=GpdConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="vae", node=VaeConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="vaegrl", node=VaeGrlConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="vaemmd", node=VaeMmdConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="frdd", node=FrddConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="dfrdd", node=DfrddConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="id", node=IdConf, package=CLF_PKG, group=CLF_GROUP)

warnings.simplefilter(action="ignore", category=RuntimeWarning)


@hydra.main(config_path="conf", config_name="ermcfg")
def launcher(hydra_config: DictConfig) -> None:
    """Instantiate with hydra and get the experiments running!"""
    cfg: ErmConfig = instantiate(hydra_config, _recursive_=True, _convert_="partial")
    run(
        cfg,
        raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True),
    )


def run(cfg: ErmConfig, raw_config: Mapping[str, str] | None = None) -> None:
    LOGGER.info(f"Current working directory: '{os.getcwd()}'")
    cfg.target = Target(cfg.target)

    logger_kwargs = dict(
        entity="predictive-analytics-lab",
        project="dev_frdd",
        offline=cfg.log_offline,
        group=cfg.model.__class__.__name__ if cfg.exp_group is None else cfg.exp_group,
    )
    train_logger = WandbLogger(**logger_kwargs, reinit=True)
    hparams = {"cwd": os.getcwd()}
    if raw_config is not None:
        LOGGER.info("-----\n" + str(raw_config) + "\n-----")
        hparams.update(raw_config)
    train_logger.log_hyperparams(hparams)

    if hasattr(cfg.data, "root"):
        cfg.data.root = to_absolute_path(cfg.data.root)
    cfg.data.setup()
    cfg.data.prepare_data()

    if hasattr(cfg.test_data, "root"):
        cfg.test_data.root = to_absolute_path(cfg.test_data.root)
    cfg.test_data.setup()
    cfg.test_data.prepare_data()

    cfg.trainer.logger = train_logger
    cfg.trainer.callbacks += []

    pl.seed_everything(cfg.seed)
    cfg.model.build(cfg.data)
    cfg.model = cfg.model.cuda()
    if cfg.model_path:
        cfg.model.load_state_dict(torch.load(cfg.model_path))
    cfg.model.freeze()

    def new_train(self: T, mode: bool = True) -> T:
        return self

    cfg.model.train = MethodType(new_train, cfg.model)

    fe = Erm(
        cfg.model,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        lr_sched_freq=cfg.model.lr_sched_freq,
        lr_sched_interval=cfg.model.lr_sched_interval,
        lr_restart_mult=cfg.model.lr_restart_mult,
        lr_initial_restart=cfg.model.lr_initial_restart,
        target=cfg.target,
    )
    cfg.trainer.fit(
        fe,
        train_dataloaders=cfg.data.train_dataloader(shuffle=True, drop_last=True),
        val_dataloaders=cfg.data.val_dataloader(),
    )
    cfg.trainer.test(fe, dataloaders=cfg.data.test_dataloader())

    # cfg.model.run(datamodule=cfg.data, trainer=cfg.trainer, seed=cfg.seed)
    # Manually invoke finish for multirun-compatibility
    train_logger.experiment.finish()


if __name__ == "__main__":
    launcher()
