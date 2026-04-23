from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.hybrid_alpr import HybridOutput


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    detector: torch.Tensor
    recognition_ctc: torch.Tensor


class HybridALPRLoss:
    def __init__(
        self,
        blank_index: int,
        detector_weight: float = 1.0,
        ctc_weight: float = 1.0,
    ) -> None:
        self.blank_index = blank_index
        self.detector_weight = detector_weight
        self.ctc_weight = ctc_weight

        self.bbox_loss_fn = nn.SmoothL1Loss()
        self.conf_loss_fn = nn.BCEWithLogitsLoss()
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)

    def __call__(
        self,
        outputs: HybridOutput,
        target_bboxes: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> LossBreakdown:
        device = outputs.ctc_logits.device
        batch_size = outputs.ctc_logits.shape[1]

        bbox_loss = self.bbox_loss_fn(outputs.bbox_pred, target_bboxes)
        confidence_target = torch.ones((batch_size, 1), device=device)
        confidence_loss = self.conf_loss_fn(outputs.confidence_logits, confidence_target)
        detector_loss = bbox_loss + confidence_loss

        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=outputs.ctc_logits.shape[0],
            dtype=torch.long,
            device=device,
        )

        ctc_loss = self.ctc_loss_fn(
            outputs.ctc_logits.log_softmax(dim=-1),
            labels,
            input_lengths,
            label_lengths,
        )

        total = self.detector_weight * detector_loss + self.ctc_weight * ctc_loss
        return LossBreakdown(total=total, detector=detector_loss, recognition_ctc=ctc_loss)
