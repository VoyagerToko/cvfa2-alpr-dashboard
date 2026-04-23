from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def character_accuracy(pred_texts: Sequence[str], true_texts: Sequence[str]) -> float:
    if not true_texts:
        return 0.0

    total_chars = 0
    correct_chars = 0
    for pred, truth in zip(pred_texts, true_texts):
        max_len = max(len(pred), len(truth))
        if max_len == 0:
            continue
        total_chars += max_len

        for i in range(max_len):
            p = pred[i] if i < len(pred) else ""
            t = truth[i] if i < len(truth) else ""
            correct_chars += int(p == t)

    if total_chars == 0:
        return 0.0
    return correct_chars / total_chars


def full_plate_accuracy(pred_texts: Sequence[str], true_texts: Sequence[str]) -> float:
    if not true_texts:
        return 0.0
    matches = sum(int(p == t) for p, t in zip(pred_texts, true_texts))
    return matches / len(true_texts)


def bbox_iou(pred_bbox: torch.Tensor, true_bbox: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU for normalized bboxes (x1, y1, x2, y2).
    """
    x1 = torch.max(pred_bbox[:, 0], true_bbox[:, 0])
    y1 = torch.max(pred_bbox[:, 1], true_bbox[:, 1])
    x2 = torch.min(pred_bbox[:, 2], true_bbox[:, 2])
    y2 = torch.min(pred_bbox[:, 3], true_bbox[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (pred_bbox[:, 2] - pred_bbox[:, 0]).clamp(min=0) * (
        pred_bbox[:, 3] - pred_bbox[:, 1]
    ).clamp(min=0)
    true_area = (true_bbox[:, 2] - true_bbox[:, 0]).clamp(min=0) * (
        true_bbox[:, 3] - true_bbox[:, 1]
    ).clamp(min=0)

    union = pred_area + true_area - inter_area
    return inter_area / (union + 1e-8)


def batch_mean_iou(pred_bbox: torch.Tensor, true_bbox: torch.Tensor) -> float:
    iou = bbox_iou(pred_bbox, true_bbox)
    return float(iou.mean().item()) if iou.numel() else 0.0


def rolling_average(values: list[float], window_size: int = 20) -> float:
    if not values:
        return 0.0
    arr = np.array(values[-window_size:], dtype=np.float32)
    return float(arr.mean())
