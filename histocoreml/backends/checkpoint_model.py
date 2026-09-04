"""Training-checkpoint segmentation backend.

Closes the loop between training and inference: a ``.pth`` written by
:class:`~histocoreml.training.trainer.SegmentationTrainer` holds a state dict,
not a serialised graph, so it needs the architecture rebuilt around it before
it can predict. :class:`TorchScriptModel` cannot load one.

The architecture and encoder come from :class:`~histocoreml.config.ModelConfig`
when set, and otherwise from the ``TrainingConfig`` the trainer embedded in the
checkpoint — so evaluating a fresh run usually needs nothing but the path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from histocoreml.backends.base_model import BaseSegmentationModel
from histocoreml.config import ModelConfig

logger = logging.getLogger(__name__)

_ARCH_ALIASES = {
    "unet": "Unet",
    "unet++": "UnetPlusPlus",
    "unetplusplus": "UnetPlusPlus",
    "deeplabv3+": "DeepLabV3Plus",
    "deeplabv3plus": "DeepLabV3Plus",
    "fpn": "FPN",
    "pspnet": "PSPNet",
}


class CheckpointModel(BaseSegmentationModel):
    """Runs a segmentation model rebuilt from a training checkpoint.

    Expected checkpoint contract: a dict with ``model_state_dict``, optionally
    ``config`` (the :class:`~histocoreml.config.TrainingConfig` used to train).
    A bare state dict also works when the config supplies architecture/encoder.

    Usage::

        cfg = ModelConfig(model_path="checkpoints/best.pth", architecture="unet++",
                          encoder="efficientnet-b4")
        with CheckpointModel(cfg) as model:
            masks = model.predict_batch(batch)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._model: nn.Module | None = None

    def load(self) -> CheckpointModel:
        model_path = Path(self._cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        logger.info("Loading checkpoint %s on %s", model_path, self._device)
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        trained_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None

        architecture = self._cfg.architecture or getattr(trained_cfg, "architecture", None)
        encoder = self._cfg.encoder or getattr(trained_cfg, "encoder", None)
        if architecture is None or encoder is None:
            raise ValueError(
                f"{model_path} does not record its architecture/encoder. Set "
                "ModelConfig(architecture=..., encoder=...) to load it."
            )

        model = self._build_model(architecture, encoder)
        model.load_state_dict(state_dict)
        self._model = model.to(self._device).eval()
        logger.info("Rebuilt %s/%s from checkpoint", architecture, encoder)
        return self

    def _build_model(self, architecture: str, encoder: str) -> nn.Module:
        """Instantiate the segmentation_models_pytorch architecture, untrained."""
        try:
            import segmentation_models_pytorch as smp  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required to load training "
                "checkpoints: pip install segmentation-models-pytorch"
            ) from exc

        key = architecture.lower()
        smp_name = _ARCH_ALIASES.get(key)
        cls = getattr(smp, smp_name) if smp_name else getattr(smp, architecture, None)
        if cls is None:
            raise ValueError(
                f"Unknown architecture '{architecture}'. Available: {sorted(_ARCH_ALIASES)}"
            )

        return cls(
            encoder_name=encoder,
            encoder_weights=None,  # weights come from the checkpoint
            in_channels=self._cfg.input_channels,
            classes=self._cfg.num_classes,
        )

    @torch.inference_mode()
    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Predict binary masks ``(N, H, W)`` for a ``(N, C, H, W)`` batch in [0, 1]."""
        probs = self.predict_proba_batch(batch)
        return (probs >= self._cfg.threshold).astype(np.uint8)

    @torch.inference_mode()
    def predict_proba_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Predict soft masks ``(N, H, W)`` with values in [0, 1]."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Use as a context manager or call load() first.")

        logits: torch.Tensor = self._model(batch.to(self._device))
        logits = self._normalise_output(logits)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    @staticmethod
    def _normalise_output(logits: torch.Tensor) -> torch.Tensor:
        """Normalise model output to shape (N, H, W)."""
        if logits.ndim == 4 and logits.shape[1] == 1:
            return logits.squeeze(1)
        if logits.ndim == 4 and logits.shape[1] > 1:
            return logits[:, 1]  # foreground channel for multi-class
        return logits
