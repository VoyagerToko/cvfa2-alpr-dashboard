from __future__ import annotations

import torch

from src.training.metrics import batch_mean_iou, character_accuracy, full_plate_accuracy


def test_character_accuracy() -> None:
    pred = ["MH12AB1234", "DL8CAF5031"]
    true = ["MH12AB1234", "DL8CAF5039"]
    score = character_accuracy(pred, true)
    assert 0.9 <= score <= 1.0


def test_full_plate_accuracy() -> None:
    pred = ["KA01AA1111", "KA01AA1112"]
    true = ["KA01AA1111", "KA01AA1113"]
    assert full_plate_accuracy(pred, true) == 0.5


def test_batch_mean_iou() -> None:
    pred_bbox = torch.tensor([[0.1, 0.1, 0.5, 0.4]], dtype=torch.float32)
    true_bbox = torch.tensor([[0.1, 0.1, 0.5, 0.4]], dtype=torch.float32)
    assert abs(batch_mean_iou(pred_bbox, true_bbox) - 1.0) < 1e-6
