"""TorchScript segmentation model backend."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from histocoreml.backends.base_model import BaseSegmentationModel
from histocoreml.config import ModelConfig

logger = logging.getLogger(__name__)


class TorchScriptModel(BaseSegmentationModel):
    """Loads and runs a TorchScript ``.pt`` segmentation model.

    Expected model contract:
    - Input:  float32 tensor ``(N, 3, H, W)`` normalised to ``[0, 1]``.
    - Output: raw logits ``(N, 1, H, W)`` or ``(N, H, W)``.
              For multi-class models the foreground channel (index 1) is used.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._model: torch.jit.ScriptModule | None = None
        self._device = torch.device(cfg.device)

    def load(self) -> TorchScriptModel:
        model_path = Path(self._cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading TorchScript model from %s on %s", model_path, self._device)
        self._model = torch.jit.load(str(model_path), map_location=self._device)
        self._model.eval()
        logger.info("Model loaded successfully")
        return self

    @torch.inference_mode()
    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Predict binary masks for a batch of patches.

        Args:
            batch: ``(N, C, H, W)`` float32 tensor in ``[0, 1]``.

        Returns:
            Binary uint8 array ``(N, H, W)``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Use as a context manager or call load() first.")

        logits: torch.Tensor = self._model(batch.to(self._device))
        logits = self._normalise_output(logits)
        probs = torch.sigmoid(logits)
        return (probs >= self._cfg.threshold).cpu().numpy().astype(np.uint8)

    @torch.inference_mode()
    def predict_proba_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Predict soft probability masks.

        Returns:
            float32 array ``(N, H, W)`` with values in [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        logits: torch.Tensor = self._model(batch.to(self._device))
        logits = self._normalise_output(logits)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy().astype(np.float32)

    def _normalise_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Normalise model output to shape (N, H, W)."""
        if logits.ndim == 4 and logits.shape[1] == 1:
            return logits.squeeze(1)
        if logits.ndim == 4 and logits.shape[1] > 1:
            return logits[:, 1]  # foreground channel for multi-class
        return logits
