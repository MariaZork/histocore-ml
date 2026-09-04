"""Model factory for creating segmentation models."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, cast

import torch.nn as nn

from histocoreml.models.deeplab import DeepLabV3Plus
from histocoreml.models.unet import UNet
from histocoreml.models.unet_plusplus import UNetPlusPlus

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type] = {
    "unet": UNet,
    "unet++": UNetPlusPlus,
    "unetplusplus": UNetPlusPlus,
    "deeplabv3+": DeepLabV3Plus,
    "deeplab": DeepLabV3Plus,
}

SUPPORTED_ENCODERS = {
    "unet": [
        "custom",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ],
    "unet++": [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ],
    "deeplabv3+": [
        "resnet50",
        "resnet101",
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
    ],
}


@dataclass
class ArchitectureConfig:
    """Configuration for building a segmentation model architecture.

    Distinct from :class:`histocoreml.config.ModelConfig`, which configures
    *inference* against already-trained weights (model path, device, patch
    size, threshold). This one describes the network to construct, and is what
    :func:`get_model` consumes.
    """

    architecture: str = "unet++"
    encoder: str = "efficientnet-b4"
    encoder_pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 1
    dropout: float = 0.0
    deep_supervision: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder_name": self.encoder,
            "encoder_pretrained": self.encoder_pretrained,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "deep_supervision": self.deep_supervision,
        }


ModelConfig = ArchitectureConfig
"""Deprecated alias for :class:`ArchitectureConfig`.

Kept so existing imports keep working. Prefer ``ArchitectureConfig`` — the old
name collided with :class:`histocoreml.config.ModelConfig`, and since both now
carry ``architecture`` and ``encoder`` fields the two are easy to mix up.
"""


def get_model(config: ArchitectureConfig | None = None, **kwargs: Any) -> nn.Module:
    """Create a segmentation model from config or keyword arguments."""
    if config is not None:
        params = config.to_dict()
        architecture = config.architecture.lower()
    else:
        params = {k: v for k, v in kwargs.items() if k != "architecture"}
        architecture = kwargs.get("architecture", "unet++").lower()

    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. Supported: {list(MODEL_REGISTRY.keys())}"
        )

    encoder_name = params.get("encoder_name", "efficientnet-b4")
    valid_encoders = SUPPORTED_ENCODERS.get(architecture, [])
    if encoder_name not in valid_encoders:
        raise ValueError(
            f"Encoder '{encoder_name}' not supported for '{architecture}'. "
            f"Supported: {valid_encoders}"
        )

    model_class = MODEL_REGISTRY[architecture]

    # Architectures accept different options (only UNet++ has deep supervision,
    # DeepLabV3+ has no dropout), so drop anything this one does not take rather
    # than raising TypeError on a valid config.
    accepted = set(inspect.signature(model_class).parameters)
    dropped = sorted(set(params) - accepted)
    if dropped:
        logger.debug("%s ignores unsupported option(s): %s", model_class.__name__, dropped)
    params = {k: v for k, v in params.items() if k in accepted}

    return model_class(**params)


def list_models() -> dict[str, list[str]]:
    """List all available models and their supported encoders."""
    return SUPPORTED_ENCODERS.copy()


def get_model_info(architecture: str) -> dict[str, Any]:
    """Get information about a specific architecture."""
    architecture = architecture.lower()
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}")

    model_class = MODEL_REGISTRY[architecture]

    return {
        "name": architecture,
        "class": model_class.__name__,
        "supported_encoders": SUPPORTED_ENCODERS.get(architecture, []),
        "docstring": model_class.__doc__,
    }


_ORGAN_CONFIGS: dict[str, dict[str, Any]] = {
    "kidney": {"num_classes": 2, "description": "Kidney glomeruli segmentation"},
    "breast": {"num_classes": 3, "description": "Breast tumor segmentation"},
    "colon": {"num_classes": 4, "description": "Colon tissue segmentation"},
    "lung": {"num_classes": 3, "description": "Lung cancer segmentation"},
    "prostate": {"num_classes": 3, "description": "Prostate gland segmentation"},
    "liver": {"num_classes": 3, "description": "Liver tissue segmentation"},
    "pancreas": {"num_classes": 3, "description": "Pancreas tissue segmentation"},
}


def create_model_for_organ(
    organ: str,
    num_classes: int | None = None,
    architecture: str = "unet++",
    encoder: str = "efficientnet-b4",
) -> nn.Module:
    """Create a pre-configured model for a specific organ type."""

    organ = organ.lower()
    if organ not in _ORGAN_CONFIGS:
        raise ValueError(f"Unknown organ: {organ}. Supported: {list(_ORGAN_CONFIGS.keys())}")

    organ_cfg = _ORGAN_CONFIGS[organ]
    num_classes = num_classes or int(cast(int, organ_cfg["num_classes"]))

    config = ArchitectureConfig(
        architecture=architecture,
        encoder=encoder,
        num_classes=num_classes,
        encoder_pretrained=True,
    )

    return get_model(config)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }
