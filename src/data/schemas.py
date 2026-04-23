from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PlateSample:
    image_path: Path
    plate_text: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    camera_id: str | None = None
    zone: str | None = None
    timestamp: str | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    @classmethod
    def from_row(cls, row: dict[str, Any], columns: dict[str, str]) -> "PlateSample":
        return cls(
            image_path=Path(str(row[columns["image_path"]])),
            plate_text=str(row[columns["plate_text"]]).strip().upper(),
            x_min=int(row[columns["x_min"]]),
            y_min=int(row[columns["y_min"]]),
            x_max=int(row[columns["x_max"]]),
            y_max=int(row[columns["y_max"]]),
            camera_id=str(row.get("camera_id", "")).strip() or None,
            zone=str(row.get("zone", "")).strip() or None,
            timestamp=str(row.get("timestamp", "")).strip() or None,
        )
