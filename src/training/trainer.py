from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.labels import LabelEncoder
from src.models.hybrid_alpr import HybridALPRModel
from src.training.losses import HybridALPRLoss
from src.training.metrics import batch_mean_iou, character_accuracy, full_plate_accuracy
from src.utils.logging_utils import get_logger


class ALPRTrainer:
    def __init__(
        self,
        model: HybridALPRModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: HybridALPRLoss,
        label_encoder: LabelEncoder,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.label_encoder = label_encoder
        self.config = config
        self.device = device

        artifacts_dir = Path(config["paths"]["artifacts_dir"])
        checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = artifacts_dir / "train_metrics.json"
        self.best_ckpt_path = checkpoints_dir / "best_model.pt"
        self.logger = get_logger("trainer")

    def fit(self) -> list[dict[str, float]]:
        epochs = int(self.config["training"]["epochs"])
        best_plate_acc = -1.0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)

            epoch_metrics = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_char_acc": val_metrics["char_acc"],
                "val_plate_acc": val_metrics["plate_acc"],
            }
            history.append(epoch_metrics)

            self.logger.info(
                (
                    "epoch=%d train_loss=%.4f val_loss=%.4f val_char_acc=%.4f "
                    "val_plate_acc=%.4f val_iou=%.4f"
                ),
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["char_acc"],
                val_metrics["plate_acc"],
                val_metrics["iou"],
            )

            if val_metrics["plate_acc"] > best_plate_acc:
                best_plate_acc = val_metrics["plate_acc"]
                self._save_checkpoint(epoch=epoch, metrics=val_metrics)

            self.metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(mode=training)

        losses: list[float] = []
        iou_scores: list[float] = []
        pred_texts_all: list[str] = []
        true_texts_all: list[str] = []

        progress = tqdm(loader, desc="train" if training else "val", leave=False)
        for batch in progress:
            images = batch["images"].to(self.device)
            plate_images = batch["plate_images"].to(self.device)
            bboxes = batch["bboxes"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)
            true_texts = batch["plate_texts"]

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images, plate_images)
            loss_pack = self.criterion(outputs, bboxes, labels, label_lengths)

            if training:
                loss_pack.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config["training"]["grad_clip_norm"])
                )
                self.optimizer.step()

            losses.append(float(loss_pack.total.detach().item()))
            iou_scores.append(batch_mean_iou(outputs.bbox_pred.detach(), bboxes.detach()))

            pred_texts = self.label_encoder.ctc_greedy_decode(outputs.ctc_logits.detach())
            pred_texts_all.extend(pred_texts)
            true_texts_all.extend(true_texts)

            progress.set_postfix(loss=sum(losses) / len(losses))

        return {
            "loss": float(sum(losses) / max(len(losses), 1)),
            "iou": float(sum(iou_scores) / max(len(iou_scores), 1)),
            "char_acc": character_accuracy(pred_texts_all, true_texts_all),
            "plate_acc": full_plate_accuracy(pred_texts_all, true_texts_all),
        }

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(payload, self.best_ckpt_path)
        self.logger.info("Saved new best checkpoint: %s", self.best_ckpt_path)
