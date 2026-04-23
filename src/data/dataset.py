from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.augmentations import build_eval_transform, build_plate_transform, build_train_transform
from src.data.cv_localization import localize_plate
from src.data.labels import LabelEncoder


class PlateDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        config: dict[str, Any],
        label_encoder: LabelEncoder,
        split: str = "train",
        use_cv_localizer: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(self.manifest_path)
        self.config = config
        self.label_encoder = label_encoder
        self.columns = config["data"]["csv_columns"]
        self.use_cv_localizer = use_cv_localizer

        self.transform = build_train_transform(config) if split == "train" else build_eval_transform(config)
        self.plate_transform = build_plate_transform(config)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = Path(str(row[self.columns["image_path"]]))
        image = self._load_image(image_path)

        x1 = int(row[self.columns["x_min"]])
        y1 = int(row[self.columns["y_min"]])
        x2 = int(row[self.columns["x_max"]])
        y2 = int(row[self.columns["y_max"]])
        bbox = [x1, y1, x2, y2]

        plate_crop = image[y1:y2, x1:x2]
        if self.use_cv_localizer:
            result = localize_plate(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), self.config["cv_localization"])
            if result.bbox is not None:
                bx1, by1, bx2, by2 = result.bbox
                bbox = [bx1, by1, bx2, by2]
            if result.crop is not None and result.crop.size > 0:
                plate_crop = cv2.cvtColor(result.crop, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image, bboxes=[bbox], bbox_labels=[1])
        image_tensor = transformed["image"]

        transformed_bbox = transformed["bboxes"][0] if transformed["bboxes"] else bbox
        tx1, ty1, tx2, ty2 = [float(v) for v in transformed_bbox]
        _, th, tw = image_tensor.shape

        bbox_tensor = torch.tensor(
            [tx1 / max(tw, 1), ty1 / max(th, 1), tx2 / max(tw, 1), ty2 / max(th, 1)], dtype=torch.float32
        )

        if plate_crop.size == 0:
            plate_crop = image
        plate_transformed = self.plate_transform(image=plate_crop)
        plate_tensor = plate_transformed["image"]

        plate_text = str(row[self.columns["plate_text"]]).strip().upper()
        label_tensor = self.label_encoder.encode(plate_text)

        return {
            "image": image_tensor,
            "plate_image": plate_tensor,
            "bbox": bbox_tensor,
            "label": label_tensor,
            "label_length": torch.tensor(len(label_tensor), dtype=torch.long),
            "plate_text": plate_text,
            "image_path": str(image_path),
        }


def plate_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    plate_images = torch.stack([item["plate_image"] for item in batch], dim=0)
    bboxes = torch.stack([item["bbox"] for item in batch], dim=0)

    labels = torch.cat([item["label"] for item in batch], dim=0)
    label_lengths = torch.stack([item["label_length"] for item in batch], dim=0)

    return {
        "images": images,
        "plate_images": plate_images,
        "bboxes": bboxes,
        "labels": labels,
        "label_lengths": label_lengths,
        "plate_texts": [item["plate_text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }
