from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(config: dict[str, Any]) -> A.Compose:
    image_h, image_w = config["data"]["image_size"]
    aug_cfg = config["preprocessing"]["augment"]
    norm_mean = config["preprocessing"]["normalize_mean"]
    norm_std = config["preprocessing"]["normalize_std"]

    return A.Compose(
        [
            A.Resize(height=image_h, width=image_w),
            A.Rotate(limit=aug_cfg.get("rotation_limit", 8), border_mode=0, p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=aug_cfg.get("blur_limit", 5), p=0.5),
                    A.GaussianBlur(blur_limit=aug_cfg.get("blur_limit", 5), p=0.5),
                ],
                p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                contrast_limit=0.2,
                p=0.5,
            ),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
    )


def build_eval_transform(config: dict[str, Any]) -> A.Compose:
    image_h, image_w = config["data"]["image_size"]
    norm_mean = config["preprocessing"]["normalize_mean"]
    norm_std = config["preprocessing"]["normalize_std"]

    return A.Compose(
        [
            A.Resize(height=image_h, width=image_w),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
    )


def build_plate_transform(config: dict[str, Any]) -> A.Compose:
    image_h, image_w = config["data"]["image_size"]
    norm_mean = config["preprocessing"]["normalize_mean"]
    norm_std = config["preprocessing"]["normalize_std"]

    return A.Compose(
        [
            A.Resize(height=max(64, image_h // 2), width=max(160, image_w // 2)),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )
