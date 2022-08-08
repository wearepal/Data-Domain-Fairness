import torch
from conduit.data import CelebADataModule, ColoredMNISTDataModule
from ranzen.hydra import Option

from dfrdd.models import (
    Ae,
    AeGrl,
    AeGrlDo,
    AeGrlEns,
    AeGrlMmd,
    Frdd,
    Gpd,
    Mmd,
    Vae,
    VaeGrl,
    VaeMmd,
)
from dfrdd.relay import DfddRelay

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    data_ops: list[Option] = [
        Option(ColoredMNISTDataModule, name="cmnist_base"),
        Option(CelebADataModule, name="celeba_base"),
    ]

    model_ops: list[Option] = [
        Option(Ae, name="ae_base"),
        Option(AeGrl, name="aegrl_base"),
        Option(AeGrlDo, name="aegrldo_base"),
        Option(AeGrlEns, name="aegrlens_base"),
        Option(AeGrlMmd, name="aegrlmmd_base"),
        Option(Mmd, name="mmd_base"),
        Option(Gpd, name="gpd_base"),
        Option(Vae, name="vae_base"),
        Option(VaeGrl, name="vaegrl_base"),
        Option(VaeMmd, name="vaemmd_base"),
        Option(Frdd, name="dfdd_base"),
    ]

    DfddRelay.with_hydra(
        root="conf",
        data=data_ops,
        model=model_ops,
        clear_cache=True,
    )
