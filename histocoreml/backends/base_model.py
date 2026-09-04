"""Abstract base class for all inference model backends.

Unified interface for segmentation, classification, embedding extraction,
and other inference tasks.
"""

from __future__ import annotations

import abc

import numpy as np
import torch

from histocoreml.config import ModelConfig


class BaseInferenceModel(abc.ABC):
    """Abstract interface for inference models.

    Concrete implementations (TorchScript, ONNX, TensorRT) must implement
    :meth:`load` and :meth:`predict_batch`.

    Supports the context-manager protocol for safe resource cleanup::

        with TorchScriptModel(cfg) as model:
            outputs = model.predict_batch(batch)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._device = torch.device(cfg.device)

    @abc.abstractmethod
    def load(self) -> BaseInferenceModel:
        """Load model weights from disk. Returns *self* for chaining."""

    @abc.abstractmethod
    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run inference on a batch of inputs.

        Args:
            batch: Input tensor (format depends on model type)
                - Segmentation: (N, C, H, W) normalized to [0, 1]
                - Classification: (N, C, H, W) normalized to [0, 1]

        Returns:
            Model outputs as numpy array
                - Segmentation: Binary uint8 array (N, H, W) with values 0 or 1
                - Classification: Float array (N, num_classes) with probabilities
        """

    def predict_proba_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Return soft probability predictions.

        Override in subclasses that support probability output.

        Returns:
            Probability array (format depends on model type)
                - Segmentation: (N, H, W) in [0, 1]
                - Classification: (N, num_classes) in [0, 1]
        """
        raise NotImplementedError(f"{type(self).__name__} does not support probability output.")

    def __enter__(self) -> BaseInferenceModel:
        return self.load()

    def __exit__(self, *_: object) -> None:  # noqa: B027
        # Intentionally a no-op: subclasses override only if they hold
        # resources. Not abstract, so simple backends need not implement it.
        pass


class BaseSegmentationModel(BaseInferenceModel):
    """Base class for segmentation inference models.

    Maintains backward compatibility with existing code.
    """

    @abc.abstractmethod
    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Predict binary masks for a batch of patches.

        Args:
            batch: (N, C, H, W) float32 tensor in [0, 1]

        Returns:
            Binary uint8 array (N, H, W) with values 0 or 1
        """

    @abc.abstractmethod
    def predict_proba_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Predict soft probability masks.

        Returns:
            float32 array (N, H, W) with values in [0, 1]
        """


# Backward compatibility alias
BaseSegmentationModel = BaseSegmentationModel
