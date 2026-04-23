from __future__ import annotations

import os
from itertools import count

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator

from src.analytics.drift import DriftMonitor
from src.analytics.occupancy import ParkingOccupancyEstimator
from src.api.dependencies import build_pipeline
from src.api.schemas import HealthResponse, ImagePredictionResponse
from src.config import load_config


CONFIG_PATH = os.getenv("APP_CONFIG", "configs/default.yaml")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
DB_PATH = os.getenv("SQLITE_DB_PATH", "artifacts/parking_events.sqlite3")
ZONE_MAP_PATH = os.getenv("DEFAULT_ZONE_MAP", "configs/zone_map.json")

config = load_config(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline, model_loaded = build_pipeline(
    config=config,
    device=device,
    checkpoint_path=MODEL_CHECKPOINT,
    db_path=DB_PATH,
    zone_map_path=ZONE_MAP_PATH,
)

frame_counter = count(1)

app = FastAPI(title="Hybrid ALPR API", version="0.1.0")
Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    camera_id: str = Form("api_cam"),
) -> ImagePredictionResponse:
    content = await file.read()
    np_data = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    event = pipeline.predict_frame(
        frame_bgr=image,
        camera_id=camera_id,
        frame_index=next(frame_counter),
        persist=True,
    )
    return ImagePredictionResponse(**event.__dict__)


@app.get("/events")
def recent_events(limit: int = 100) -> dict[str, object]:
    if pipeline.event_store is None:
        return {"events": []}
    return {"events": pipeline.event_store.get_recent_events(limit=limit)}


@app.get("/analytics/occupancy")
def occupancy() -> dict[str, object]:
    if pipeline.event_store is None:
        return {"occupancy": {}, "total_active": 0}

    estimator = ParkingOccupancyEstimator(
        store=pipeline.event_store,
        timeout_seconds=int(config["analytics"]["occupancy_timeout_seconds"]),
    )
    by_zone = estimator.estimate_current_occupancy()
    return {
        "occupancy": by_zone,
        "total_active": int(sum(by_zone.values())),
    }


@app.get("/analytics/drift")
def drift(baseline_accuracy: float, current_accuracy: float, threshold: float = 0.05) -> dict[str, object]:
    monitor = DriftMonitor(threshold_absolute_drop=threshold)
    report = monitor.evaluate(
        baseline_accuracy=baseline_accuracy,
        current_accuracy=current_accuracy,
    )
    return report.__dict__
