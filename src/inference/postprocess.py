from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import Levenshtein


@dataclass(slots=True)
class PlateObservation:
    plate_text: str
    timestamp: datetime
    plate_hash: str


def normalize_plate_text(plate_text: str) -> str:
    return "".join(ch for ch in plate_text.upper().strip() if ch.isalnum())


def plate_hash(plate_text: str) -> str:
    normalized = normalize_plate_text(plate_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DuplicateDetector:
    def __init__(self, window_seconds: int = 20, levenshtein_threshold: int = 1) -> None:
        self.window = timedelta(seconds=window_seconds)
        self.levenshtein_threshold = levenshtein_threshold
        self._history: deque[PlateObservation] = deque()

    def is_duplicate(self, plate_text: str, timestamp: datetime | None = None) -> bool:
        timestamp = timestamp or datetime.utcnow()
        current_text = normalize_plate_text(plate_text)
        current_hash = plate_hash(current_text)

        while self._history and (timestamp - self._history[0].timestamp) > self.window:
            self._history.popleft()

        for obs in self._history:
            if obs.plate_hash == current_hash:
                return True
            if Levenshtein.distance(current_text, obs.plate_text) <= self.levenshtein_threshold:
                return True

        self._history.append(
            PlateObservation(plate_text=current_text, timestamp=timestamp, plate_hash=current_hash)
        )
        return False
