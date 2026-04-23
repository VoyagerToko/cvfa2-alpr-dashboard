from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import ensure_directories, load_config
from src.data.dataset import PlateDataset, plate_collate_fn
from src.data.labels import LabelEncoder
from src.models.hybrid_alpr import HybridALPRModel
from src.training.losses import HybridALPRLoss
from src.training.trainer import ALPRTrainer
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid ALPR model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)
    set_global_seed(int(config["seed"]))

    processed_dir = Path(config["paths"]["processed_dir"])
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "Missing train/val splits. Run: python scripts/prepare_data.py --config configs/default.yaml"
        )

    label_encoder = LabelEncoder(config["model"]["vocab"])

    train_ds = PlateDataset(train_csv, config, label_encoder=label_encoder, split="train")
    val_ds = PlateDataset(val_csv, config, label_encoder=label_encoder, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        collate_fn=plate_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        collate_fn=plate_collate_fn,
    )

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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    criterion = HybridALPRLoss(
        blank_index=label_encoder.blank_index,
        detector_weight=float(config["training"]["detector_loss_weight"]),
        ctc_weight=float(config["training"]["ctc_loss_weight"]),
    )

    trainer = ALPRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        label_encoder=label_encoder,
        config=config,
        device=device,
    )

    history = trainer.fit()
    print("[train] Completed training")
    print(f"[train] Last epoch metrics: {history[-1] if history else {}}")


if __name__ == "__main__":
    main()
