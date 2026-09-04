"""SegmentationTrainer — training loop for histology segmentation models.

Refactored to use BaseSegmentationTrainer abstract base class for consistency
with the rest of the HistoCoreML architecture.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from histocoreml.config import TrainerConfig, TrainingConfig
from histocoreml.training.base_trainer import BaseSegmentationTrainer
from histocoreml.training.losses import get_loss
from histocoreml.training.metrics import dice_score, iou_score
from histocoreml.utils.progress import progress_bar

logger = logging.getLogger(__name__)


class SegmentationTrainer(BaseSegmentationTrainer[TrainerConfig]):
    """Training loop for patch-level segmentation models.

    Supports:
    - Dice, BCE, Dice+BCE, Focal, Tversky losses
    - AdamW / SGD optimisers with cosine annealing LR
    - Automatic Mixed Precision (AMP)
    - Early stopping
    - Best checkpoint saving

    Inherits from :class:`~histocoreml.training.base_trainer.BaseSegmentationTrainer`
    for consistent API across all trainers.

    Usage::

        cfg     = TrainingConfig(architecture="unet", encoder="resnet50", epochs=100)
        trainer = SegmentationTrainer(cfg)
        trainer.fit(train_loader, val_loader)

    Or with context manager::

        with SegmentationTrainer(cfg) as trainer:
            history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        model: nn.Module | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            cfg: Training configuration
            model: Optional pre-built model. If None, model will be built on first access.
            criterion: Optional pre-built loss. If None, one is built from
                ``cfg.loss`` with default hyper-parameters. Pass this to use a
                loss configured with custom weights, e.g.
                ``get_loss("dice_bce", dice_weight=0.7, bce_weight=0.3)``.
        """
        # Convert TrainingConfig to base TrainerConfig attributes
        # output_dir is the run root; checkpoint_dir is explicit so that
        # BaseTrainer does not append another "checkpoints" segment to it.
        base_config = TrainerConfig(
            output_dir=cfg.checkpoint_dir.parent,
            checkpoint_dir=cfg.checkpoint_dir,
            experiment_name=cfg.experiment_name,
            seed=42,  # Default seed
            device=cfg.device if hasattr(cfg, "device") else "auto",
            mixed_precision=cfg.mixed_precision,
            log_level="INFO",
        )

        # Store the full training config
        self._training_cfg = cfg

        super().__init__(base_config)

        # Set model if provided
        if model is not None:
            self.model = model

        # Initialize AMP scaler
        if torch.__version__ >= "2.0":
            self._scaler = torch.amp.GradScaler(
                device=str(self.device),
                enabled=(cfg.mixed_precision and self.device.type == "cuda"),
            )
        else:
            self._scaler = torch.cuda.amp.GradScaler(
                enabled=(cfg.mixed_precision and self.device.type == "cuda")
            )

        # Initialize training components (lazy initialization)
        self.criterion = criterion if criterion is not None else self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

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
            "unet": smp.Unet,
            "unet++": smp.UnetPlusPlus,
            "deeplabv3+": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "segformer": smp.Segformer if hasattr(smp, "Segformer") else smp.Unet,
        }

        arch = self._training_cfg.architecture.lower()
        cls = arch_map.get(arch, smp.Unet)

        return cls(
            encoder_name=self._training_cfg.encoder,
            encoder_weights="imagenet" if self._training_cfg.pretrained else None,
            in_channels=3,
            classes=1,
        )

    def _build_criterion(self) -> nn.Module:
        """Build the loss function."""
        return get_loss(self._training_cfg.loss)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer."""
        if self._training_cfg.optimizer.lower() == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self._training_cfg.learning_rate,
                momentum=0.9,
                weight_decay=self._training_cfg.weight_decay,
            )
        return AdamW(
            self.model.parameters(),
            lr=self._training_cfg.learning_rate,
            weight_decay=self._training_cfg.weight_decay,
        )

    def _build_scheduler(self) -> Any:
        """Build the learning rate scheduler."""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self._training_cfg.epochs,
            eta_min=1e-6,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Args:
            train_loader: DataLoader for training patches.
            val_loader:   DataLoader for validation patches.
            on_epoch_end: Optional callback(epoch, metrics_dict) called after each epoch.
                          Metrics include: train_loss, val_dice, val_iou, lr.

        Returns:
            Dict with training history (train_loss, val_dice per epoch).
        """
        self.state = self.state.replace(is_training=True)

        # Initialize history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_dice": [],
            "val_iou": [],
            "learning_rate": [],
        }
        self.state = self.state.replace(history=history)

        for epoch in range(1, self._training_cfg.epochs + 1):
            self.state = self.state.replace(epoch=epoch)
            t0 = time.perf_counter()

            # Training phase
            train_loss = self._train_epoch(train_loader)

            # Validation phase
            val_metrics = self._validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Update history
            history = self.state.history.copy()
            history["train_loss"].append(train_loss)
            history["val_dice"].append(val_metrics["dice"])
            history["val_iou"].append(val_metrics["iou"])
            history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            self.state = self.state.replace(history=history)

            elapsed = time.perf_counter() - t0
            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_dice=%.4f | val_iou=%.4f | %.1fs",
                epoch,
                self._training_cfg.epochs,
                train_loss,
                val_metrics["dice"],
                val_metrics["iou"],
                elapsed,
            )

            # Log to TensorBoard
            self._log_scalar("train/loss", train_loss, epoch)
            self._log_scalar("val/dice", val_metrics["dice"], epoch)
            self._log_scalar("val/iou", val_metrics["iou"], epoch)
            self._log_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

            # Call optional epoch-end callback
            if on_epoch_end is not None:
                metrics = {
                    "train_loss": train_loss,
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                on_epoch_end(epoch, metrics)

            # Checkpoint saving
            if val_metrics["dice"] > self.state.best_metric:
                self.state = self.state.replace(best_metric=val_metrics["dice"], patience_counter=0)
                self._save_checkpoint(epoch, val_metrics["dice"], is_best=True)
            else:
                new_counter = self.state.patience_counter + 1
                self.state = self.state.replace(patience_counter=new_counter)
                if new_counter >= self._training_cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch,
                        self._training_cfg.early_stopping_patience,
                    )
                    break

        logger.info("Training complete. Best val Dice: %.4f", self.state.best_metric)
        self.state = self.state.replace(is_training=False)

        # Save final history
        self.save_history()

        return self.state.history

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        with progress_bar(loader, desc=f"Epoch {self.state.epoch}", unit="batch") as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass with AMP
                autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.amp.autocast(
                    device_type=autocast_device,
                    enabled=(self._training_cfg.mixed_precision and self.device.type == "cuda"),
                ):
                    preds = self.model(images)
                    loss = self.criterion(preds.squeeze(1), masks.squeeze(1))

                # Backward pass
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()

                total_loss += loss.item()
                num_batches += 1

                # Update global step
                self.state = self.state.replace(global_step=self.state.global_step + 1)

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.inference_mode()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Run validation.

        Args:
            loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        dices, ious = [], []

        with progress_bar(loader, desc="Validation", unit="batch") as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].cpu().numpy().astype(np.uint8).squeeze(1)

                # Forward pass
                autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.amp.autocast(
                    device_type=autocast_device,
                    enabled=(self._training_cfg.mixed_precision and self.device.type == "cuda"),
                ):
                    preds = torch.sigmoid(self.model(images)).cpu().numpy().squeeze(1)

                binary = (preds >= 0.5).astype(np.uint8)
                batch_dices = []
                for p, m in zip(binary, masks, strict=False):
                    d = dice_score(p, m)
                    dices.append(d)
                    batch_dices.append(d)
                    ious.append(iou_score(p, m))

                # Update progress bar with running average
                pbar.set_postfix({"dice": f"{np.mean(batch_dices):.4f}"})

        return {
            "dice": float(np.mean(dices) if dices else 0.0),
            "iou": float(np.mean(ious) if ious else 0.0),
        }

    def _save_checkpoint(self, epoch: int, val_dice: float, is_best: bool = False) -> None:
        """Save a checkpoint.

        Args:
            epoch: Current epoch number
            val_dice: Validation Dice score
            is_best: Whether this is the best checkpoint so far
        """
        path = self.checkpoint_dir / f"best_epoch{epoch:03d}_dice{val_dice:.4f}.pth"

        extra_data = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_dice": val_dice,
            "config": self._training_cfg,
        }

        self.save_checkpoint(path, extra_data)

        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            self.save_checkpoint(best_path, extra_data)
            logger.info("Best checkpoint saved (Dice: %.4f)", val_dice)
