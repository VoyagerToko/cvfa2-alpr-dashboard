from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    last_frame_index: int


class CentroidTracker:
    def __init__(self, max_distance: float = 90.0, max_inactive_frames: int = 30) -> None:
        self.max_distance = max_distance
        self.max_inactive_frames = max_inactive_frames
        self.next_track_id = 1
        self.tracks: dict[int, TrackState] = {}

    @staticmethod
    def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, detections: list[tuple[int, int, int, int]], frame_index: int) -> list[int]:
        assigned_ids: list[int] = []

        for det_bbox in detections:
            det_centroid = np.array(self._centroid(det_bbox), dtype=np.float32)

            best_track_id = None
            best_distance = float("inf")

            for track_id, track in self.tracks.items():
                track_centroid = np.array(track.centroid, dtype=np.float32)
                dist = float(np.linalg.norm(det_centroid - track_centroid))
                if dist < best_distance and dist <= self.max_distance:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            else:
                track_id = best_track_id

            self.tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=det_bbox,
                centroid=tuple(det_centroid.tolist()),
                last_frame_index=frame_index,
            )
            assigned_ids.append(track_id)

        to_delete = [
            tid
            for tid, track in self.tracks.items()
            if frame_index - track.last_frame_index > self.max_inactive_frames
        ]
        for tid in to_delete:
            del self.tracks[tid]

        return assigned_ids
