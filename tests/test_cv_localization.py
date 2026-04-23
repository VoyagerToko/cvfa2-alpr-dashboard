from __future__ import annotations

import cv2
import numpy as np

from src.data.cv_localization import detect_harris_corners, localize_plate


def test_detect_harris_corners_returns_points() -> None:
    image = np.zeros((120, 240), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (200, 80), 255, 2)

    points = detect_harris_corners(image)
    assert points.shape[1] == 2


def test_localize_plate_on_synthetic_image() -> None:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 90), (300, 140), (255, 255, 255), -1)

    cfg = {
        "threshold_value": 180,
        "threshold_max": 255,
        "canny_low": 80,
        "canny_high": 180,
        "min_plate_area_ratio": 0.005,
        "max_plate_area_ratio": 0.3,
        "enable_perspective_correction": True,
    }

    result = localize_plate(image, cfg)
    assert result.bbox is not None
    assert result.crop is not None
