from types import MethodType
from typing import Callable, Dict, NamedTuple, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from ranzen import implements
from torch import Tensor, nn
from torchvision.models import vgg19


class VggOut(NamedTuple):
    # block5_conv2: torch.Tensor
    # block1_conv1: torch.Tensor
    # block2_conv1: torch.Tensor
    block3_conv1: torch.Tensor
    block4_conv1: torch.Tensor
    block5_conv1: torch.Tensor
    pool5: Optional[torch.Tensor] = None


T = TypeVar("T")


class VGG(nn.Module):
    """Torchvision VGG Model.

    features:
    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace=True)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace=True)
      (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): ReLU(inplace=True)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace=True)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace=True)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace=True)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )

    avgpool:
    Sequential()

    classifier:
    Sequential(
      (0): Linear(in_features=25088, out_features=4096, bias=True))
      (1): ReLU(inplace=True))
      (2): Dropout(p=0.5, inplace=False))
      (3): Linear(in_features=4096, out_features=4096, bias=True))
      (4): ReLU(inplace=True))
      (5): Dropout(p=0.5, inplace=False))
      (6): Linear(in_features=4096, out_features=1000, bias=True))
    )
    """

    def __init__(self, output_layers: Dict[str, int]) -> None:
        super().__init__()
        self.model = vgg19(pretrained=True)
        self.model.training = False
        self.model.requires_grad_(False)

        self.model.eval()

        def new_train(self: T, mode: bool = True) -> T:
            return self

        self.model.train = MethodType(new_train, self.model)

        self.output_layers = output_layers
        self.selected_out: Dict[str, Tensor] = {}
        self.fhooks = []

        for layer_name, layer in self.output_layers.items():
            self.fhooks.append(
                self.model.features[layer].register_forward_hook(
                    self.forward_hook(layer_name)
                )
            )
        self.fhooks.append(
            self.model.avgpool.register_forward_hook(self.forward_hook("pool5"))
        )

    def forward_hook(
        self, layer_name: str
    ) -> Callable[[nn.Module, Tensor, Tensor], None]:
        """Add a forward hook."""

        def hook(module: nn.Module, input: Tensor, output: Tensor) -> None:
            self.selected_out[layer_name] = F.normalize(output.flatten(start_dim=1))

        return hook

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tuple[Tensor, VggOut]:
        _ = self.model(x)
        return VggOut(**self.selected_out)
