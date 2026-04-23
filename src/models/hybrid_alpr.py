from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.detector import PlateDetector
from src.models.recognizer import PlateRecognizer


@dataclass(slots=True)
class HybridOutput:
    bbox_pred: torch.Tensor
    confidence_logits: torch.Tensor
    ctc_logits: torch.Tensor


class HybridALPRModel(nn.Module):
    def __init__(
        self,
        detector_backbone: str,
        recognizer_backbone: str,
        vocab_size: int,
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_layers: int,
        transformer_ff_dim: int,
        dropout: float = 0.1,
        pretrained_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.detector = PlateDetector(detector_backbone, pretrained=pretrained_backbones)
        self.recognizer = PlateRecognizer(
            backbone_name=recognizer_backbone,
            vocab_size=vocab_size,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
            pretrained=pretrained_backbones,
        )

    def forward(self, images: torch.Tensor, plate_images: torch.Tensor) -> HybridOutput:
        bbox_pred, confidence_logits = self.detector(images)
        ctc_logits = self.recognizer(plate_images)
        return HybridOutput(
            bbox_pred=bbox_pred,
            confidence_logits=confidence_logits,
            ctc_logits=ctc_logits,
        )
