from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class LocalizationResult:
    bbox: tuple[int, int, int, int] | None
    corners: np.ndarray | None
    crop: np.ndarray | None
    score: float


def threshold_segmentation(image: np.ndarray, threshold_value: int = 180, threshold_max: int = 255) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, threshold_value, threshold_max, cv2.THRESH_BINARY)
    return binary


def region_growing(binary_image: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    h, w = binary_image.shape[:2]
    x_seed, y_seed = seed_point
    if not (0 <= x_seed < w and 0 <= y_seed < h):
        return np.zeros_like(binary_image)
    if binary_image[y_seed, x_seed] == 0:
        return np.zeros_like(binary_image)

    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque([(x_seed, y_seed)])

    while queue:
        x, y = queue.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = 1

        if binary_image[y, x] == 0:
            continue

        mask[y, x] = 255

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                if binary_image[ny, nx] > 0:
                    queue.append((nx, ny))

    return mask


def edge_based_segmentation(image: np.ndarray, canny_low: int = 80, canny_high: int = 180) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / max(h, 1)
        if area > 120 and 1.8 <= aspect <= 6.5:
            boxes.append((x, y, x + w, y + h))
    return boxes


def detect_harris_corners(image: np.ndarray, max_corners: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray_float = np.float32(gray)

    harris_response = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    harris_response = cv2.dilate(harris_response, None)

    threshold = 0.01 * float(harris_response.max())
    ys, xs = np.where(harris_response > threshold)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    responses = harris_response[ys, xs]
    ranked_idx = np.argsort(responses)[::-1][:max_corners]
    points = np.stack([xs[ranked_idx], ys[ranked_idx]], axis=1).astype(np.float32)
    return points


def _find_segmentation_bbox(
    binary: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = binary.shape[:2]
    image_area = h * w
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        ratio = area / max(image_area, 1)
        aspect = cw / max(ch, 1)
        if min_area_ratio <= ratio <= max_area_ratio and 1.8 <= aspect <= 6.5:
            score = ratio * 2.0 + (1.0 - abs(aspect - 4.0) / 4.0)
            candidates.append((score, (x, y, x + cw, y + ch)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    ordered = _order_points(corners)
    (tl, tr, br, bl) = ordered

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    max_width = max(1, max_width)
    max_height = max(1, max_height)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def localize_plate(image: np.ndarray, cfg: dict[str, Any]) -> LocalizationResult:
    h, w = image.shape[:2]
    binary = threshold_segmentation(
        image,
        threshold_value=int(cfg.get("threshold_value", 180)),
        threshold_max=int(cfg.get("threshold_max", 255)),
    )

    min_ratio = float(cfg.get("min_plate_area_ratio", 0.005))
    max_ratio = float(cfg.get("max_plate_area_ratio", 0.2))

    bbox = _find_segmentation_bbox(binary, min_ratio, max_ratio)

    if bbox is None:
        edge_boxes = edge_based_segmentation(
            image,
            canny_low=int(cfg.get("canny_low", 80)),
            canny_high=int(cfg.get("canny_high", 180)),
        )
        if edge_boxes:
            bbox = sorted(edge_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[0]

    if bbox is None:
        return LocalizationResult(bbox=None, corners=None, crop=None, score=0.0)

    x1, y1, x2, y2 = bbox
    seed = ((x1 + x2) // 2, (y1 + y2) // 2)
    grown_mask = region_growing(binary, seed)

    if int(grown_mask.sum()) == 0:
        grown_mask = np.zeros(binary.shape, dtype=np.uint8)
        grown_mask[y1:y2, x1:x2] = 255

    masked_region = cv2.bitwise_and(image, image, mask=grown_mask)
    corners = detect_harris_corners(masked_region)

    candidate_corners = corners[
        (corners[:, 0] >= x1)
        & (corners[:, 0] <= x2)
        & (corners[:, 1] >= y1)
        & (corners[:, 1] <= y2)
    ]

    plate_crop: np.ndarray
    if len(candidate_corners) >= 4:
        rect = cv2.minAreaRect(candidate_corners.reshape(-1, 1, 2))
        box = cv2.boxPoints(rect).astype(np.float32)
        if cfg.get("enable_perspective_correction", True):
            plate_crop = _warp_perspective(image, box)
        else:
            plate_crop = image[y1:y2, x1:x2]
        corners_out = box
    else:
        plate_crop = image[y1:y2, x1:x2]
        corners_out = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    score = min(1.0, (len(candidate_corners) / 50.0) + (bbox_area / max(1, h * w)))

    return LocalizationResult(bbox=(x1, y1, x2, y2), corners=corners_out, crop=plate_crop, score=float(score))


def draw_localization(image: np.ndarray, result: LocalizationResult) -> np.ndarray:
    output = image.copy()
    if result.bbox:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if result.corners is not None and len(result.corners) >= 4:
        for point in result.corners.astype(int):
            cv2.circle(output, tuple(point), 3, (0, 0, 255), -1)
    return output
