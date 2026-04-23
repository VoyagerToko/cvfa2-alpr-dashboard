from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.analytics.storage import EventStore
from src.data.cv_localization import localize_plate
from src.data.labels import LabelEncoder
from src.inference.postprocess import DuplicateDetector, normalize_plate_text
from src.inference.tracker import CentroidTracker
from src.models.hybrid_alpr import HybridALPRModel


@dataclass(slots=True)
class InferenceEvent:
    timestamp: str
    plate_text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    camera_id: str
    zone: str
    track_id: int
    is_duplicate: bool


class ALPRInferencePipeline:
    def __init__(
        self,
        model: HybridALPRModel,
        label_encoder: LabelEncoder,
        config: dict[str, Any],
        device: torch.device,
        db_path: str | None = None,
        zone_map_path: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.label_encoder = label_encoder
        self.config = config
        self.device = device

        infer_cfg = config["inference"]
        self.duplicate_detector = DuplicateDetector(
            window_seconds=int(infer_cfg["duplicate_window_seconds"]),
            levenshtein_threshold=int(infer_cfg["duplicate_levenshtein_threshold"]),
        )
        self.tracker = CentroidTracker(max_distance=float(infer_cfg["tracker_max_distance"]))

        self.event_store = EventStore(db_path) if db_path else None

        self.zone_map = self._load_zone_map(zone_map_path)
        self.default_zone = str(config["analytics"].get("default_zone", "UNKNOWN"))

        self.image_h, self.image_w = config["data"]["image_size"]
        self.mean = np.array(config["preprocessing"]["normalize_mean"], dtype=np.float32)
        self.std = np.array(config["preprocessing"]["normalize_std"], dtype=np.float32)

    @staticmethod
    def _load_zone_map(zone_map_path: str | None) -> dict[str, str]:
        if not zone_map_path:
            return {}
        path = Path(zone_map_path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _prepare_tensor(self, image_rgb: np.ndarray, target_hw: tuple[int, int]) -> torch.Tensor:
        h, w = target_hw
        resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _bbox_from_normalized(self, bbox_norm: np.ndarray, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        x1 = int(np.clip(bbox_norm[0], 0.0, 1.0) * frame_w)
        y1 = int(np.clip(bbox_norm[1], 0.0, 1.0) * frame_h)
        x2 = int(np.clip(bbox_norm[2], 0.0, 1.0) * frame_w)
        y2 = int(np.clip(bbox_norm[3], 0.0, 1.0) * frame_h)

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        if x2 == x1:
            x2 = min(frame_w, x1 + 1)
        if y2 == y1:
            y2 = min(frame_h, y1 + 1)

        return x1, y1, x2, y2

    @staticmethod
    def _safe_crop(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(1, min(w, x2))
        y2 = max(1, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return image_bgr
        return image_bgr[y1:y2, x1:x2]

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        frame_index: int,
        timestamp: datetime | None = None,
        persist: bool = True,
    ) -> InferenceEvent:
        timestamp = timestamp or datetime.utcnow()
        frame_h, frame_w = frame_bgr.shape[:2]

        loc_result = localize_plate(frame_bgr, self.config["cv_localization"])

        if loc_result.bbox is not None:
            bbox = loc_result.bbox
            detector_confidence = float(loc_result.score)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detector_input = self._prepare_tensor(frame_rgb, (self.image_h, self.image_w)).to(self.device)
            with torch.no_grad():
                bbox_pred, confidence_logits = self.model.detector(detector_input)
            detector_confidence = float(torch.sigmoid(confidence_logits).squeeze().cpu().item())
            bbox = self._bbox_from_normalized(bbox_pred.squeeze(0).cpu().numpy(), frame_w, frame_h)

        if loc_result.crop is not None and loc_result.crop.size > 0:
            plate_bgr = loc_result.crop
        else:
            plate_bgr = self._safe_crop(frame_bgr, bbox)

        plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        plate_h = max(64, self.image_h // 2)
        plate_w = max(160, self.image_w // 2)
        plate_tensor = self._prepare_tensor(plate_rgb, (plate_h, plate_w)).to(self.device)

        with torch.no_grad():
            ctc_logits = self.model.recognizer(plate_tensor)
            probs = torch.softmax(ctc_logits, dim=-1)
            token_confidence = float(probs.max(dim=-1).values.mean().cpu().item())

        plate_text = self.label_encoder.ctc_greedy_decode(ctc_logits)[0]
        plate_text = normalize_plate_text(plate_text)
        is_duplicate = self.duplicate_detector.is_duplicate(plate_text, timestamp=timestamp)

        track_ids = self.tracker.update([bbox], frame_index=frame_index)
        track_id = track_ids[0] if track_ids else -1

        zone = self.zone_map.get(camera_id, self.default_zone)
        final_confidence = float((detector_confidence + token_confidence) / 2.0)

        event = InferenceEvent(
            timestamp=timestamp.isoformat(),
            plate_text=plate_text,
            confidence=final_confidence,
            bbox=bbox,
            camera_id=camera_id,
            zone=zone,
            track_id=track_id,
            is_duplicate=is_duplicate,
        )

        if persist and self.event_store is not None:
            self.event_store.insert_event(asdict(event))

        return event

    @staticmethod
    def annotate_frame(frame_bgr: np.ndarray, event: InferenceEvent) -> np.ndarray:
        output = frame_bgr.copy()
        x1, y1, x2, y2 = event.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{event.plate_text} | Z:{event.zone} | TID:{event.track_id}"
        cv2.putText(output, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return output
