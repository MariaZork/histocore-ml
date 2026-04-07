"""ONNX Runtime segmentation model backend.

Provides cross-platform, hardware-accelerated inference without requiring
a full PyTorch installation on the deployment target.

Install the runtime::

    pip install onnxruntime          # CPU
    pip install onnxruntime-gpu      # GPU (CUDA)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from histocoreml.config import ModelConfig
from histocoreml.inference.base_model import BaseSegmentationModel

logger = logging.getLogger(__name__)


class ONNXModel(BaseSegmentationModel):
    """Segmentation model backend using ONNX Runtime.

    Usage::

        cfg   = ModelConfig(model_path="model.onnx", device="cpu")
        with ONNXModel(cfg) as model:
            masks = model.predict_batch(batch_tensor)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._session: Any | None = None
        self._input_name: str | None = None

    def load(self) -> ONNXModel:
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required: pip install onnxruntime"
            ) from exc

        model_path = Path(self._cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = self._get_providers()
        logger.info("Loading ONNX model from %s with providers: %s", model_path, providers)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNX model loaded successfully")
        return self

    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Model not loaded.")

        np_batch = batch.cpu().numpy().astype(np.float32)
        assert self._session is not None
        assert self._input_name is not None
        outputs = self._session.run(None, {self._input_name: np_batch})
        logits = outputs[0]

        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim == 4 and logits.shape[1] > 1:
            logits = logits[:, 1]

        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        return (probs >= self._cfg.threshold).astype(np.uint8)

    def _get_providers(self) -> list[str]:
        """Select ONNX Runtime execution providers based on config device."""
        device = self._cfg.device.lower()
        if "cuda" in device:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "mps" in device:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
