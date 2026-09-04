"""HistoCoreML segmentation models.

Supported models:
- UNet: Classic U-Net with various backbones
- UNet++: Nested U-Net with dense skip connections
- DeepLabV3+: Atrous spatial pyramid pooling
- SAM: Segment Anything Model adaptation

Usage::

    from histocoreml.models import ArchitectureConfig, get_model

    cfg = ArchitectureConfig(architecture="unet++", encoder="efficientnet-b4", num_classes=3)
    model = get_model(cfg)

Note the two similarly-named configs: :class:`ArchitectureConfig` (here)
describes a network to *build*; :class:`histocoreml.config.ModelConfig`
configures *inference* against trained weights.
"""

from histocoreml.models.deeplab import DeepLabV3Plus
from histocoreml.models.factory import (
    ArchitectureConfig,
    ModelConfig,
    count_parameters,
    create_model_for_organ,
    get_model,
    get_model_info,
    list_models,
)
from histocoreml.models.unet import UNet
from histocoreml.models.unet_plusplus import UNetPlusPlus

__all__ = [
    # Architectures
    "UNet",
    "UNetPlusPlus",
    "DeepLabV3Plus",
    # Factory
    "ArchitectureConfig",
    "ModelConfig",  # deprecated alias for ArchitectureConfig
    "get_model",
    "get_model_info",
    "list_models",
    "create_model_for_organ",
    "count_parameters",
]
