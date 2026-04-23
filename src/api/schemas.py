from __future__ import annotations

from pydantic import BaseModel, Field


class ImagePredictionResponse(BaseModel):
    timestamp: str
    plate_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int]
    camera_id: str
    zone: str
    track_id: int
    is_duplicate: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
