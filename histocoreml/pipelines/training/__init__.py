"""Training pipelines for HistoCoreML.

End-to-end training workflows including:
- On-the-fly patch extraction from whole-slide images
- Model training
- Validation
- Checkpointing
- Optional inference on a held-out directory

:class:`SegmentationTrainingPipeline` is driven by an
:class:`~histocoreml.config.ExperimentConfig`, so a whole run is described by a
single YAML document.

To train from tiles already on disk, use
:func:`~histocoreml.training.build_train_dataloader` with
:class:`~histocoreml.training.SegmentationTrainer` directly — that is what
``histo-train --images/--masks`` does. :func:`extract_patches_to_disk` writes
those tiles from whole-slide images in the layout that pairing expects.
"""

from histocoreml.pipelines.training.segmentation import (
    ExtractionStats,
    SegmentationTrainingPipeline,
    extract_patches_to_disk,
)

__all__ = [
    "SegmentationTrainingPipeline",
    "extract_patches_to_disk",
    "ExtractionStats",
]
