from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.labels import LabelEncoder
from src.inference.pipeline import ALPRInferencePipeline
from src.models.hybrid_alpr import HybridALPRModel


def build_model(config: dict[str, Any], vocab_size: int) -> HybridALPRModel:
    model_cfg = config["model"]
    return HybridALPRModel(
        detector_backbone=model_cfg["detector_backbone"],
        recognizer_backbone=model_cfg["recognizer_backbone"],
        vocab_size=vocab_size,
        transformer_d_model=int(model_cfg["transformer_d_model"]),
        transformer_nhead=int(model_cfg["transformer_nhead"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
        transformer_ff_dim=int(model_cfg["transformer_ff_dim"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained_backbones=False,
    )


def load_checkpoint_if_available(
    model: HybridALPRModel,
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> bool:
    if checkpoint_path is None:
        return False

    path = Path(checkpoint_path)
    if not path.exists():
        return False

    payload = torch.load(path, map_location=device)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=False)
    return True


def build_pipeline(
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: str | Path | None,
    db_path: str,
    zone_map_path: str,
) -> tuple[ALPRInferencePipeline, bool]:
    label_encoder = LabelEncoder(config["model"]["vocab"])
    model = build_model(config, vocab_size=label_encoder.vocab_size)
    ckpt_loaded = load_checkpoint_if_available(model, checkpoint_path, device)

    pipeline = ALPRInferencePipeline(
        model=model,
        label_encoder=label_encoder,
        config=config,
        device=device,
        db_path=db_path,
        zone_map_path=zone_map_path,
    )
    return pipeline, ckpt_loaded
