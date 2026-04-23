from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


PLATE_REGEX = re.compile(r"^[A-Z0-9-]{4,16}$")


def verify_manifest(
    manifest_path: str | Path,
    columns: dict[str, str],
    strict_plate_pattern: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate annotation rows and return (clean_df, errors).
    """
    df = pd.read_csv(manifest_path)
    errors: list[str] = []
    valid_rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        image_path = Path(str(row[columns["image_path"]]))
        plate_text = str(row[columns["plate_text"]]).strip().upper()

        if not image_path.exists():
            errors.append(f"row={idx}: image not found -> {image_path}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            errors.append(f"row={idx}: failed to read image -> {image_path}")
            continue

        h, w = img.shape[:2]
        x_min = int(row[columns["x_min"]])
        y_min = int(row[columns["y_min"]])
        x_max = int(row[columns["x_max"]])
        y_max = int(row[columns["y_max"]])

        bbox_valid = 0 <= x_min < x_max <= w and 0 <= y_min < y_max <= h
        if not bbox_valid:
            errors.append(
                f"row={idx}: invalid bbox ({x_min},{y_min},{x_max},{y_max}) for image size ({w},{h})"
            )
            continue

        if strict_plate_pattern and not PLATE_REGEX.match(plate_text):
            errors.append(f"row={idx}: invalid plate text -> {plate_text}")
            continue

        normalized_row = row.to_dict()
        normalized_row[columns["plate_text"]] = plate_text
        valid_rows.append(normalized_row)

    clean_df = pd.DataFrame(valid_rows)
    return clean_df, errors
