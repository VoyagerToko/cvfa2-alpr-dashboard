from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import cv2
import torch

from src.config import load_config
from src.data.labels import LabelEncoder
from src.inference.pipeline import ALPRInferencePipeline
from src.models.hybrid_alpr import HybridALPRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time ALPR on video")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--camera-id", type=str, default="entry_cam_1")
    parser.add_argument("--output-video", type=str, default="")
    parser.add_argument("--save-events", type=str, default="artifacts/inference_events.csv")
    return parser.parse_args()


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> tuple[HybridALPRModel, LabelEncoder]:
    label_encoder = LabelEncoder(config["model"]["vocab"])
    model_cfg = config["model"]
    model = HybridALPRModel(
        detector_backbone=model_cfg["detector_backbone"],
        recognizer_backbone=model_cfg["recognizer_backbone"],
        vocab_size=label_encoder.vocab_size,
        transformer_d_model=int(model_cfg["transformer_d_model"]),
        transformer_nhead=int(model_cfg["transformer_nhead"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
        transformer_ff_dim=int(model_cfg["transformer_ff_dim"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"], strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()
    return model, label_encoder


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, label_encoder = load_model(config, Path(args.checkpoint), device)
    pipeline = ALPRInferencePipeline(
        model=model,
        label_encoder=label_encoder,
        config=config,
        device=device,
        db_path="artifacts/parking_events.sqlite3",
        zone_map_path="configs/zone_map.json",
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    events_out = Path(args.save_events)
    events_out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        event = pipeline.predict_frame(
            frame_bgr=frame,
            camera_id=args.camera_id,
            frame_index=frame_index,
            persist=True,
        )
        rows.append(asdict(event))

        annotated = pipeline.annotate_frame(frame, event)
        if writer is not None:
            writer.write(annotated)

    cap.release()
    if writer is not None:
        writer.release()

    if rows:
        with events_out.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(rows)

    print(f"[infer_video] Processed frames: {frame_index}")
    print(f"[infer_video] Events saved at: {events_out}")


if __name__ == "__main__":
    main()
