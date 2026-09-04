"""Factory for foundation model encoders."""

from __future__ import annotations

from histocoreml.config import FoundationConfig
from histocoreml.foundation.base_encoder import BaseEncoder
from histocoreml.foundation.vit_encoder import UNIEncoder, ViTEncoder


def get_encoder(cfg: FoundationConfig) -> BaseEncoder:
    """Return an encoder instance for the specified foundation model.

    Args:
        cfg: Foundation model configuration.

    Returns:
        An uninitialised :class:`BaseEncoder` subclass.

    Raises:
        ValueError: If the model_name is not recognised.
    """
    name = cfg.model_name.lower()

    if name == "uni":
        return UNIEncoder(cfg)

    if name in ("vit", "custom"):
        timm_name = "vit_large_patch16_224"
        return ViTEncoder(cfg, timm_name=timm_name)

    if name == "ctranspath":
        return ViTEncoder(cfg, timm_name="swin_tiny_patch4_window7_224")

    raise ValueError(
        f"Unknown foundation model '{name}'. "
        "Supported: uni, vit, custom, ctranspath. "
        "For CONCH/PLIP, load weights as ViTEncoder with model_path."
    )
