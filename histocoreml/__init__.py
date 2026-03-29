"""HistoCoreML — Production ML framework for computational histology.

Covers the full spectrum of computational pathology tasks:

* **WSI I/O**          — OpenSlide-backed readers for SVS, TIFF, MRXS, NDPI and more
* **Segmentation**     — Patch-based inference pipeline with concurrent tiling
* **Foundation models**— Encoders (UNI, CONCH, PLIP, custom ViT) for feature extraction
* **Training**         — Dataset builders, augmentation, loss functions, trainers
* **Biomarkers**       — Cell detection, nuclei segmentation, spatial-graph features
* **Output**           — TIFF, NPY, RLE (plain & COCO), GeoJSON, Zarr writers

Quick-start::

    from histocoreml.config import PipelineConfig
    from histocoreml.pipeline import SegmentationPipeline
    from pathlib import Path

    cfg      = PipelineConfig.from_yaml("configs/default.yaml")
    pipeline = SegmentationPipeline(cfg)
    results  = pipeline.run([Path("slide.svs")])

"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("histocoreml")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
