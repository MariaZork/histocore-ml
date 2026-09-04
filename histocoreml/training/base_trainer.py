"""Abstract base class for all trainers in HistoCoreML.

This module defines the core interface that all trainers must implement,
providing a consistent API for training segmentation models, foundation models,
and other histology ML tasks.

Usage::

    from histocoreml.training.base_trainer import BaseTrainer, TrainingState

    class MyCustomTrainer(BaseTrainer):
        def _build_model(self) -> nn.Module:
            # Implementation
            pass

        def fit(self) -> dict[str, list[float]]:
            # Implementation
            pass
"""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from histocoreml.config import TrainerConfig, TrainingState
from histocoreml.utils.seed import seed_everything

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=TrainerConfig)


class BaseTrainer(abc.ABC, Generic[T]):
    """Abstract base class for all trainers.

    Provides a common interface and shared functionality for training
    different types of models in histology ML. Concrete implementations
    must implement :meth:`fit` and :meth:`_build_model`.

    Supports the context-manager protocol for safe resource cleanup::

        with MyTrainer(config) as trainer:
            trainer.fit()

    Attributes:
        config: Trainer configuration
        state: Current training state
        device: Torch device being used
        model: The neural network model (initialized on first access or in fit())
    """

    def __init__(self, config: T) -> None:
        self.config = config
        self.state = TrainingState()
        self.device = torch.device(config.device)
        self._model: nn.Module | None = None
        self._writer: SummaryWriter | None = None

        # Setup output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(config.checkpoint_dir or config.output_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.config.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Set seed for reproducibility
        self._set_seed(config.seed)

        logger.info(f"Initialized {self.__class__.__name__} (device: {self.device})")

    @property
    def model(self) -> nn.Module:
        """Get the model, building it if necessary."""
        if self._model is None:
            self._model = self._build_model()
            self._model = self._model.to(self.device)
        return self._model

    @model.setter
    def model(self, value: nn.Module) -> None:
        """Set the model directly."""
        self._model = value.to(self.device)

    @abc.abstractmethod
    def _build_model(self) -> nn.Module:
        """Build and return the neural network model.

        This method is called lazily when the model is first accessed.
        Implementations should not move the model to device - this is
        handled automatically by the property getter.

        Returns:
            Uninitialized model instance
        """

    @abc.abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> dict[str, list[float]]:
        """Run the training loop.

        This is the main entry point for training. Implementations should:
        1. Initialize data loaders
        2. Initialize optimizer and scheduler
        3. Run training epochs
        4. Validate periodically
        5. Save checkpoints
        6. Update training state

        Returns:
            Training history dictionary with metrics per epoch
        """

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        seed_everything(seed)

    def _get_writer(self) -> SummaryWriter | None:
        """Get or create TensorBoard writer."""
        if self._writer is None and not self.state.is_training:
            self._writer = SummaryWriter(self.log_dir)
        return self._writer

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard."""
        writer = self._get_writer()
        if writer is not None:
            writer.add_scalar(tag, value, step)

    def save_checkpoint(self, path: Path | str, extra: dict[str, Any] | None = None) -> None:
        """Write model weights and training state to *path*.

        The layout is the one :meth:`load_checkpoint` reads back, so any
        checkpoint written here can resume training. Callers add run-specific
        entries (optimizer state, the metric that triggered the save) via
        *extra*.

        Args:
            path:  Destination file. Parent directories are created.
            extra: Additional keys merged into the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_metric": self.state.best_metric,
            "patience_counter": self.state.patience_counter,
            "history": self.state.history,
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        logger.debug("Saved checkpoint → %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model and training state from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model state
        if self._model is None:
            self._model = self._build_model()
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self.device)

        # Load training state
        self.state = TrainingState(
            epoch=checkpoint.get("epoch", 0),
            global_step=checkpoint.get("global_step", 0),
            best_metric=checkpoint.get("best_metric", 0.0),
            patience_counter=checkpoint.get("patience_counter", 0),
            history=checkpoint.get("history", {}),
        )

        logger.info(f"Loaded checkpoint from {path} (epoch {self.state.epoch})")

    def save_history(self) -> None:
        """Save training history to JSON."""
        history_file = self.config.output_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.state.history, f, indent=2)
        logger.info(f"Training history saved to {history_file}")

    def __enter__(self) -> BaseTrainer[T]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - cleanup resources."""
        if self._writer is not None:
            self._writer.close()


# Alias for backward compatibility
BaseSegmentationTrainer = BaseTrainer
