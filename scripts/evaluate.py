from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import PlateDataset, plate_collate_fn
from src.data.labels import LabelEncoder
from src.models.hybrid_alpr import HybridALPRModel
from src.training.metrics import batch_mean_iou, character_accuracy, full_plate_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid ALPR model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    return parser.parse_args()


def build_model(config: dict[str, Any], label_encoder: LabelEncoder, checkpoint_path: Path, device: torch.device) -> HybridALPRModel:
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
    return model


def evaluate_loader(
    model: HybridALPRModel,
    loader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> dict[str, float]:
    pred_texts: list[str] = []
    true_texts: list[str] = []
    iou_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            plate_images = batch["plate_images"].to(device)
            bboxes = batch["bboxes"].to(device)

            outputs = model(images, plate_images)
            pred_texts.extend(label_encoder.ctc_greedy_decode(outputs.ctc_logits))
            true_texts.extend(batch["plate_texts"])
            iou_scores.append(batch_mean_iou(outputs.bbox_pred, bboxes))

    return {
        "character_accuracy": character_accuracy(pred_texts, true_texts),
        "full_plate_accuracy": full_plate_accuracy(pred_texts, true_texts),
        "mean_iou": float(sum(iou_scores) / max(len(iou_scores), 1)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    label_encoder = LabelEncoder(config["model"]["vocab"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, label_encoder, Path(args.checkpoint), device)

    processed_dir = Path(config["paths"]["processed_dir"])
    test_csv = processed_dir / "test.csv"
    test_ds = PlateDataset(test_csv, config, label_encoder=label_encoder, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        collate_fn=plate_collate_fn,
    )

    results = {"test": evaluate_loader(model, test_loader, label_encoder, device)}

    real_world_manifest = Path(config["paths"]["real_world_eval_dir"]) / "manifest.csv"
    if real_world_manifest.exists():
        real_ds = PlateDataset(real_world_manifest, config, label_encoder=label_encoder, split="test")
        real_loader = DataLoader(
            real_ds,
            batch_size=int(config["data"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=bool(config["data"]["pin_memory"]),
            collate_fn=plate_collate_fn,
        )
        results["real_world_unseen"] = evaluate_loader(model, real_loader, label_encoder, device)

    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "evaluation_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"[evaluate] Saved: {out_path}")


if __name__ == "__main__":
    main()
