"""Abstract base class for all segmentation model backends."""

from __future__ import annotations

import abc

import numpy as np
import torch

from histocoreml.config import ModelConfig


class BaseSegmentationModel(abc.ABC):
    """Abstract interface for a segmentation model.

    Concrete implementations (TorchScript, ONNX, TensorRT, HuggingFace)
    must implement :meth:`load` and :meth:`predict_batch`.

    Supports the context-manager protocol for safe resource cleanup::

        with TorchScriptModel(cfg) as model:
            masks = model.predict_batch(batch)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg

    @abc.abstractmethod
    def load(self) -> BaseSegmentationModel:
        """Load model weights from disk. Returns *self* for chaining."""

    @abc.abstractmethod
    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run inference on a batch of patches.

        Args:
            batch: Float32 tensor of shape (N, C, H, W) normalised to [0, 1].

        Returns:
            Binary uint8 array of shape (N, H, W) with values 0 or 1.
        """

    def predict_proba_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Return soft probability predictions (N, H, W) in [0, 1].

        Override in subclasses that support probability output.
        Default implementation applies threshold to get binary masks and
        raises NotImplementedError — subclasses should override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support probability output."
        )

    def __enter__(self) -> BaseSegmentationModel:
        return self.load()

    def __exit__(self, *_: object) -> None:
        pass
