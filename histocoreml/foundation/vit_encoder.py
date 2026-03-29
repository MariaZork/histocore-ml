"""Generic Vision Transformer encoder backed by timm.

Can load any timm-compatible ViT variant with custom or ImageNet weights.
Also serves as the backbone for UNI, CONCH (when weights are available locally).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from histocoreml.config import FoundationConfig
from histocoreml.foundation.base_encoder import BaseEncoder

logger = logging.getLogger(__name__)

# ImageNet statistics used by most ViT models
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# UNI-specific mean/std (trained on pathology data)
_UNI_MEAN = (0.70322989, 0.53606487, 0.66096631)
_UNI_STD  = (0.21716536, 0.26081574, 0.20723780)


class ViTEncoder(BaseEncoder):
    """Generic ViT patch encoder via timm.

    Args:
        cfg:         FoundationConfig specifying model name, weights path, etc.
        timm_name:   timm model identifier (e.g. ``'vit_large_patch16_224'``).
        mean, std:   Normalisation statistics. Defaults to ImageNet.
        use_cls_token: If True, return the [CLS] token; else mean-pool patch tokens.

    Usage::

        cfg = FoundationConfig(
            model_name="custom",
            model_path=Path("uni_weights.pth"),
            embedding_dim=1024,
            patch_size=224,
        )
        with ViTEncoder(cfg, timm_name="vit_large_patch16_224") as enc:
            embeddings = enc.encode_batch(batch)
    """

    def __init__(
        self,
        cfg: FoundationConfig,
        timm_name: str = "vit_large_patch16_224",
        mean: tuple = _IMAGENET_MEAN,
        std: tuple = _IMAGENET_STD,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__(cfg)
        self._timm_name = timm_name
        self._mean = mean
        self._std = std
        self._use_cls_token = use_cls_token

    def load(self) -> "ViTEncoder":
        try:
            import timm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "timm is required for ViTEncoder: pip install histocoreml[foundation]"
            ) from exc

        logger.info("Loading timm model: %s on %s", self._timm_name, self._device)
        self._model = timm.create_model(
            self._timm_name,
            pretrained=self._cfg.model_path is None,
            num_classes=0,   # remove classification head → returns embeddings
        )

        if self._cfg.model_path is not None:
            ckpt_path = Path(self._cfg.model_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Model weights not found: {ckpt_path}")
            logger.info("Loading custom weights from %s", ckpt_path)
            state = torch.load(str(ckpt_path), map_location="cpu")
            # Handle various checkpoint formats
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state, strict=False)

        self._model = self._model.to(self._device).eval()
        logger.info("Encoder ready: dim=%d", self._cfg.embedding_dim)
        return self

    @torch.inference_mode()
    def encode_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Encode a batch of patches.

        Args:
            batch: Float32 tensor (N, 3, H, W) with values in [0, 1].

        Returns:
            float32 array (N, embedding_dim).
        """
        if self._model is None:
            raise RuntimeError("Encoder not loaded. Use as context manager or call load().")

        # Apply normalisation on-device
        mean = torch.tensor(self._mean, device=self._device).view(1, 3, 1, 1)
        std  = torch.tensor(self._std,  device=self._device).view(1, 3, 1, 1)
        x = (batch.to(self._device) - mean) / std

        out = self._model(x)

        # timm returns (N, D) with num_classes=0; handle if tuple
        if isinstance(out, (tuple, list)):
            out = out[0]

        return out.cpu().numpy().astype(np.float32)

    def get_transform(self):
        """Return a minimal torchvision transform (normalisation already in encode_batch)."""
        try:
            from torchvision import transforms  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("torchvision is required.") from exc

        return transforms.Compose([
            transforms.Resize((self._cfg.patch_size, self._cfg.patch_size)),
            transforms.ToTensor(),
        ])


class UNIEncoder(ViTEncoder):
    """UNI foundation model encoder (MedIBL / HMS).

    Requires model weights downloaded from HuggingFace:
    ``hf_hub_download(repo_id="MahmoodLab/UNI", filename="pytorch_model.bin")``

    Usage::

        cfg = FoundationConfig(model_name="uni", model_path=Path("uni.pth"),
                               embedding_dim=1024, target_mpp=0.5)
        with UNIEncoder(cfg) as enc:
            embeddings = enc.encode_batch(batch)
    """

    def __init__(self, cfg: FoundationConfig) -> None:
        super().__init__(
            cfg,
            timm_name="vit_large_patch16_224",
            mean=_UNI_MEAN,
            std=_UNI_STD,
            use_cls_token=True,
        )
