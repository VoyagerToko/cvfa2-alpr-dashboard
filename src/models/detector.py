from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn_backbone import build_cnn_backbone


class PlateDetector(nn.Module):
    """
    Lightweight single-box detector for number plate localization.
    Predicts normalized bbox (x1, y1, x2, y2) and confidence.
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = False) -> None:
        super().__init__()
        self.backbone, channels = build_cnn_backbone(backbone_name, pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bbox_head = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        pooled = self.pool(features).flatten(1)

        bbox_pred = self.bbox_head(pooled)
        confidence_logits = self.confidence_head(pooled)
        return bbox_pred, confidence_logits
