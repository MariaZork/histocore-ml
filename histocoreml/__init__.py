"""HistoCoreML — Production ML framework for computational histology.

Covers the full spectrum of computational pathology tasks:

* **WSI I/O**          — OpenSlide-backed readers for SVS, TIFF, MRXS, NDPI and more
* **Segmentation**     — Patch-based inference pipeline with concurrent tiling
* **Foundation models**— Encoders (UNI, CONCH, PLIP, custom ViT) for feature extraction
* **Training**         — Dataset builders, augmentation, loss functions, trainers
* **Biomarkers**       — Cell detection, nuclei segmentation, spatial-graph features
* **Output**           — TIFF, NPY, RLE (plain & COCO), GeoJSON, Zarr writers
* **Models**           — UNet++, UNet, DeepLabV3+ with EfficientNet/ResNet backbones

Quick-start::

    from histocoreml.config import SegmentationPipelineConfig
    from histocoreml.pipelines import create_pipeline
    from pathlib import Path

    cfg      = SegmentationPipelineConfig.from_yaml("configs/default.yaml")
    pipeline = create_pipeline("segmentation", cfg)
    results  = pipeline.run([Path("slide.svs")])

New in v0.2.0::

    # UNet++ with EfficientNet backbone
    from histocoreml.models import get_model, ArchitectureConfig
    config = ArchitectureConfig(architecture="unet++", encoder="efficientnet-b4")
    model = get_model(config)

    # End-to-end training pipeline, driven by one YAML document
    from histocoreml.config import ExperimentConfig
    from histocoreml.pipelines import SegmentationTrainingPipeline

    cfg = ExperimentConfig.from_yaml("configs/hubmap_glomeruli.yaml")
    pipeline = SegmentationTrainingPipeline(cfg)
    result = pipeline.run()  # Tiles slides on the fly → trains → validates → infers
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("histocoreml")
except PackageNotFoundError:
    __version__ = "0.2.0-dev"

__all__ = ["__version__"]

# Optional convenience exports. These need torch and segmentation-models-pytorch,
# so a core-only install skips them rather than failing at import time. The
# warning matters: without it a genuine bug in these modules looks identical to
# a missing optional dependency.
try:
    from histocoreml.models import (  # noqa: F401  (re-exported below)
        ArchitectureConfig,
        ModelConfig,
        create_model_for_organ,
        get_model,
        list_models,
    )

    __all__.extend(
        [
            "get_model",
            "ArchitectureConfig",
            "ModelConfig",
            "create_model_for_organ",
            "list_models",
        ]
    )
except ImportError as exc:  # pragma: no cover - depends on install extras
    _logging.getLogger(__name__).debug("Model factory exports unavailable: %s", exc)

# RLE utilities (moved from deprecated datasets module)
try:
    from histocoreml.output.rle_codec import rle_decode, rle_encode  # noqa: F401

    __all__.extend(["rle_decode", "rle_encode"])
except ImportError:
    pass
