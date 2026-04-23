from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torchvision.models as models


def build_cnn_backbone(name: str, pretrained: bool = False) -> Tuple[nn.Module, int]:
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        backbone = nn.Sequential(*list(model.children())[:-2])
        out_channels = 512
        return backbone, out_channels

    if name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        backbone = nn.Sequential(*list(model.children())[:-2])
        out_channels = 512
        return backbone, out_channels

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        backbone = model.features
        out_channels = 1280
        return backbone, out_channels

    raise ValueError(f"Unsupported backbone: {name}")
