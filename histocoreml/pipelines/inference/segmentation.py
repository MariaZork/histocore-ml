"""Segmentation inference pipeline.

Segmentation inference pipeline using unified base classes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from histocoreml.backends.base_model import BaseInferenceModel
from histocoreml.backends.factory import get_inference_model
from histocoreml.config import SegmentationInferenceResult, SegmentationPipelineConfig
from histocoreml.io.factory import get_reader
from histocoreml.output.factory import get_writer
from histocoreml.output.overlay_writer import save_overlay as _save_overlay
from histocoreml.pipelines.base import BaseInferencePipeline
from histocoreml.postprocessing.mask_assembler import MaskAssembler
from histocoreml.preprocessing.grid_generator import generate_patch_coords
from histocoreml.preprocessing.patch_dataset import build_dataloader
from histocoreml.utils.progress import progress_bar

logger = logging.getLogger(__name__)


class SegmentationInferencePipeline(
    BaseInferencePipeline[SegmentationPipelineConfig, SegmentationInferenceResult]
):
    """End-to-end WSI segmentation inference pipeline.

    Usage::

        from histocoreml.config import SegmentationInferenceResult, SegmentationPipelineConfig
        from histocoreml.pipelines import SegmentationInferencePipeline

        cfg = SegmentationPipelineConfig.from_yaml("configs/hubmap_glomeruli.yaml")
        pipeline = SegmentationInferencePipeline(cfg)
        results = pipeline.run([Path("slide.svs")])
    """

    def __init__(self, cfg: SegmentationPipelineConfig) -> None:
        super().__init__(cfg)
        self._model: BaseInferenceModel | None = None

    def _load_model(self) -> BaseInferenceModel:
        """Build the inference backend named by ``config.model.backend``."""
        return get_inference_model(self.config.model)

    def process_slide(self, path: Path) -> SegmentationInferenceResult:
        """Process a single slide."""
        logger.info("=" * 60)
        logger.info("Segmentation: %s", path.name)

        with get_reader(path) as reader:
            metadata = reader.get_metadata()
            coords = generate_patch_coords(
                metadata, self.config.model, self.config.tiling, slide_id=path.stem
            )

        if not coords:
            raise ValueError(f"No patch coordinates generated for {path.name}")

        logger.info(
            "Slide: %d × %d px | mpp: %s | %d patches",
            *metadata.dimensions,
            f"{metadata.mpp:.4f}" if metadata.mpp else "unknown",
            len(coords),
        )

        # Create assembler
        assembler = MaskAssembler(
            metadata=metadata,
            model_cfg=self.config.model,
            tiling_cfg=self.config.tiling,
            downsample_factor=self.config.output.downsample_factor or 1,
        )

        patch_count = 0
        total_batches = (
            len(coords) + self.config.model.batch_size - 1
        ) // self.config.model.batch_size

        # Process patches
        with self._load_model() as model, get_reader(path) as reader:
            loader = build_dataloader(
                slide_path=path,
                coords=coords,
                tiling_cfg=self.config.tiling,
                model_patch_size=self.config.model.patch_size,
                batch_size=self.config.model.batch_size,
                normalise=self.config.model.stain_normalise,
            )

            with progress_bar(loader, total=total_batches, desc=path.stem) as pbar:
                for batch in pbar:
                    if batch is None:
                        continue
                    masks: np.ndarray = model.predict_batch(batch["images"])
                    assembler.add_batch(masks, batch["coords"])
                    patch_count += len(batch["coords"])

        # Finalize and save
        logger.info("Finalising mask...")
        final_mask = assembler.finalise()

        writer = get_writer(self.config.output)
        write_result = writer.write(mask=final_mask, metadata=metadata, stem=path.stem)
        assembler.cleanup()

        # Generate overlay if requested
        if self.config.output.save_overlay:
            with get_reader(path) as reader:
                _save_overlay(
                    slide_path=path,
                    mask=final_mask,
                    stem=path.stem,
                    cfg=self.config.output,
                    reader=reader,
                )

        logger.info(
            "Done: %s | %d patches | mask → %s",
            path.name,
            patch_count,
            write_result.path,
        )

        return SegmentationInferenceResult(
            wsi_path=path,
            patch_count=patch_count,
            write_result=write_result,
            mask_path=write_result.path,
        )

    def _create_error_result(self, path: Path, error: Exception) -> SegmentationInferenceResult:
        """Create an error result."""
        return SegmentationInferenceResult(
            wsi_path=Path(path),
            elapsed_seconds=0.0,
            errors=[str(error)],
            patch_count=0,
            write_result=None,
            mask_path=None,
        )


def create_segmentation_pipeline(
    cfg: SegmentationPipelineConfig,
) -> SegmentationInferencePipeline:
    """Create a SegmentationInferencePipeline instance.

    Args:
        cfg: Pipeline configuration

    Returns:
        Configured SegmentationInferencePipeline

    Example::

        from histocoreml.config import SegmentationInferenceResult, SegmentationPipelineConfig
        from histocoreml.pipelines.inference import create_segmentation_pipeline

        cfg = SegmentationPipelineConfig.from_yaml("configs/hubmap_glomeruli.yaml")
        pipeline = create_segmentation_pipeline(cfg)
        results = pipeline.run([Path("slide.svs")])
    """
    return SegmentationInferencePipeline(cfg)
