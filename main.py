from conduit.data import CelebADataModule, ColoredMNISTDataModule
from ranzen.hydra import Option

from dfrdd.models import Frdd
from dfrdd.relay import DfddRelay

if __name__ == "__main__":
    data_ops: list[Option] = [
        Option(ColoredMNISTDataModule, name="cmnist_base"),
        Option(CelebADataModule, name="celeba_base"),
    ]

    model_ops: list[Option] = [Option(Frdd, name="dfdd_base")]

    DfddRelay.with_hydra(
        root="conf",
        data=data_ops,
        model=model_ops,
        clear_cache=True,
    )
