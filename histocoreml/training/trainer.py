"""SegmentationTrainer — training loop for histology segmentation models."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from histocoreml.config import TrainingConfig
from histocoreml.training.losses import get_loss
from histocoreml.training.metrics import dice_score, iou_score

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Training loop for patch-level segmentation models.

    Supports:
    - Dice, BCE, Dice+BCE, Focal, Tversky losses
    - AdamW / SGD optimisers with cosine annealing LR
    - Automatic Mixed Precision (AMP)
    - Early stopping
    - Best checkpoint saving

    Usage::

        cfg     = TrainingConfig(architecture="unet", encoder="resnet50", epochs=100)
        trainer = SegmentationTrainer(cfg)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(self, cfg: TrainingConfig, model: Optional[nn.Module] = None) -> None:
        self._cfg = cfg
        self._model = model or self._build_model()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._criterion = get_loss(cfg.loss)
        self._optimizer = self._build_optimizer()
        self._scheduler = CosineAnnealingLR(self._optimizer, T_max=cfg.epochs, eta_min=1e-6)
        self._scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and self._device.type == "cuda")
        self._best_dice: float = 0.0
        self._no_improve: int = 0
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Run the training loop.

        Args:
            train_loader: DataLoader for training patches.
            val_loader:   DataLoader for validation patches.

        Returns:
            Dict with training history (train_loss, val_dice per epoch).
        """
        history: Dict[str, List[float]] = {"train_loss": [], "val_dice": [], "val_iou": []}

        for epoch in range(1, self._cfg.epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_epoch(train_loader)
            val_dice, val_iou = self._val_epoch(val_loader)
            self._scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_dice"].append(val_dice)
            history["val_iou"].append(val_iou)

            elapsed = time.perf_counter() - t0
            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_dice=%.4f | val_iou=%.4f | %.1fs",
                epoch, self._cfg.epochs, train_loss, val_dice, val_iou, elapsed,
            )

            if val_dice > self._best_dice:
                self._best_dice = val_dice
                self._no_improve = 0
                self._save_checkpoint(epoch, val_dice)
            else:
                self._no_improve += 1
                if self._no_improve >= self._cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch, self._cfg.early_stopping_patience,
                    )
                    break

        logger.info("Training complete. Best val Dice: %.4f", self._best_dice)
        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(self._device)
            masks  = batch["mask"].to(self._device)
            self._optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self._cfg.mixed_precision and self._device.type == "cuda"):
                preds = self._model(images)
                loss  = self._criterion(preds.squeeze(1), masks.squeeze(1))
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.inference_mode()
    def _val_epoch(self, loader: DataLoader):
        self._model.eval()
        dices, ious = [], []
        for batch in loader:
            images = batch["image"].to(self._device)
            masks  = batch["mask"].cpu().numpy().astype(np.uint8).squeeze(1)
            preds  = torch.sigmoid(self._model(images)).cpu().numpy().squeeze(1)
            binary = (preds >= 0.5).astype(np.uint8)
            for p, m in zip(binary, masks):
                dices.append(dice_score(p, m))
                ious.append(iou_score(p, m))
        return float(np.mean(dices) if dices else 0.0), float(np.mean(ious) if ious else 0.0)

    def _save_checkpoint(self, epoch: int, val_dice: float) -> None:
        path = self._cfg.checkpoint_dir / f"best_epoch{epoch:03d}_dice{val_dice:.4f}.pth"
        torch.save({
            "epoch": epoch, "val_dice": val_dice,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "config": self._cfg,
        }, str(path))
        logger.info("Checkpoint saved → %s", path)

    def _build_model(self) -> nn.Module:
        """Build a segmentation model using segmentation_models_pytorch."""
        try:
            import segmentation_models_pytorch as smp  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for training: "
                "pip install segmentation-models-pytorch"
            ) from exc

        arch_map = {
            "unet":       smp.Unet,
            "unet++":     smp.UnetPlusPlus,
            "deeplabv3+": smp.DeepLabV3Plus,
            "segformer":  smp.Segformer if hasattr(smp, "Segformer") else smp.Unet,
        }
        cls = arch_map.get(self._cfg.architecture.lower(), smp.Unet)
        return cls(
            encoder_name=self._cfg.encoder,
            encoder_weights="imagenet" if self._cfg.pretrained else None,
            in_channels=3,
            classes=1,
        )

    def _build_optimizer(self):
        if self._cfg.optimizer.lower() == "sgd":
            return SGD(self._model.parameters(), lr=self._cfg.learning_rate,
                       momentum=0.9, weight_decay=self._cfg.weight_decay)
        return AdamW(self._model.parameters(), lr=self._cfg.learning_rate,
                     weight_decay=self._cfg.weight_decay)
