from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn_backbone import build_cnn_backbone
from src.models.transformer_decoder import TransformerSequenceModel


class PlateRecognizer(nn.Module):
    """
    CNN feature extractor + Transformer sequence model for character prediction.
    Uses CTC-compatible output logits.
    """

    def __init__(
        self,
        backbone_name: str,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone, channels = build_cnn_backbone(backbone_name, pretrained=pretrained)

        self.sequence_model = TransformerSequenceModel(
            input_dim=channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            vocab_size=vocab_size,
        )

    def forward(self, plate_images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(plate_images)  # (B, C, H, W)
        features = features.mean(dim=2)  # average pool over height -> (B, C, W)
        features = features.permute(0, 2, 1)  # (B, T, C)
        return self.sequence_model(features)
