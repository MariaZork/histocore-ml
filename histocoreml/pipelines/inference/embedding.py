"""Embedding/feature extraction inference pipeline.

Refactored from foundation/factory.py to use unified base classes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from histocoreml.config import (
    EmbeddingInferenceResult,
    FoundationConfig,
    ModelConfig,
    PipelineConfig,
    TilingConfig,
)
from histocoreml.foundation import get_encoder
from histocoreml.foundation.base_encoder import BaseEncoder
from histocoreml.io.factory import get_reader
from histocoreml.pipelines.base import BaseInferencePipeline
from histocoreml.preprocessing.grid_generator import generate_patch_coords
from histocoreml.preprocessing.patch_dataset import build_dataloader

logger = logging.getLogger(__name__)


class EmbeddingInferencePipeline(BaseInferencePipeline[PipelineConfig, EmbeddingInferenceResult]):
    """WSI-level feature extraction pipeline.

    Extracts patch embeddings using foundation models.

    Usage::

        from histocoreml.config import FoundationConfig
        from histocoreml.pipelines import EmbeddingInferencePipeline

        cfg = FoundationConfig(model_name="uni", target_mpp=0.5)
        encoder = get_encoder(cfg)
        pipeline = EmbeddingInferencePipeline(cfg, encoder)
        results = pipeline.run([Path("slide.svs")], output_dir=Path("embeddings"))
    """

    def __init__(
        self,
        foundation_cfg: FoundationConfig,
        encoder: BaseEncoder | None = None,
    ) -> None:
        # Create base pipeline config
        pipeline_cfg = PipelineConfig(
            log_level="INFO",
            output_dir=Path("embeddings"),
            num_workers=4,
            device=foundation_cfg.device,
        )
        super().__init__(pipeline_cfg)

        self._foundation_cfg = foundation_cfg
        self._encoder = encoder or get_encoder(foundation_cfg)

    def process_slide(self, path: Path) -> EmbeddingInferenceResult:
        """Process a single slide and extract embeddings."""
        logger.info("=" * 60)
        logger.info("Embedding extraction: %s", path.name)

        # Configure for embedding extraction
        model_cfg = ModelConfig(
            model_path=path,
            patch_size=self._foundation_cfg.patch_size,
            target_mpp=self._foundation_cfg.target_mpp,
            batch_size=self._foundation_cfg.batch_size,
            device=self._foundation_cfg.device,
        )
        tiling_cfg = TilingConfig(overlap=0, tissue_threshold=0.05)

        with get_reader(path) as reader:
            metadata = reader.get_metadata()
            coords = generate_patch_coords(metadata, model_cfg, tiling_cfg, slide_id=path.stem)

        loader = build_dataloader(
            slide_path=path,
            coords=coords,
            tiling_cfg=tiling_cfg,
            model_patch_size=self._foundation_cfg.patch_size,
            batch_size=self._foundation_cfg.batch_size,
        )

        all_emb: list[np.ndarray] = []
        all_coords = []

        with self._encoder as enc:
            for batch in loader:
                if batch is None:
                    continue
                emb = enc.encode_batch_normalised(batch["images"])
                all_emb.append(emb)
                all_coords.extend(batch["coords"])

        embeddings = (
            np.concatenate(all_emb, axis=0)
            if all_emb
            else np.empty((0, self._foundation_cfg.embedding_dim), dtype=np.float32)
        )

        logger.info("Done: %s | %d patches embedded", path.name, len(all_coords))

        return EmbeddingInferenceResult(
            wsi_path=path,
            patch_count=len(all_coords),
            embeddings=embeddings,
            coords=all_coords,
            model_name=self._foundation_cfg.model_name,
        )

    def run(
        self,
        wsi_paths: list[Path],
        output_dir: Path | None = None,
        save: bool = True,
    ) -> list[EmbeddingInferenceResult]:
        """Run embedding extraction with optional saving.

        Args:
            wsi_paths: List of WSI files
            output_dir: Directory to save embeddings
            save: Whether to save embeddings to disk

        Returns:
            List of embedding results
        """
        results = super().run(wsi_paths)

        if save and output_dir:
            for result in results:
                if result.success:
                    result.save(output_dir)

        return results

    def _create_error_result(self, path: Path, error: Exception) -> EmbeddingInferenceResult:
        """Create an error result."""
        return EmbeddingInferenceResult(
            wsi_path=Path(path),
            elapsed_seconds=0.0,
            errors=[str(error)],
            patch_count=0,
            embeddings=np.empty((0, 0), dtype=np.float32),
            coords=[],
        )


def create_embedding_pipeline(
    cfg: FoundationConfig,
    encoder: BaseEncoder | None = None,
) -> EmbeddingInferencePipeline:
    """Create an EmbeddingInferencePipeline instance.

    Args:
        cfg: Foundation model configuration
        encoder: Optional encoder (created from cfg if None)

    Returns:
        Configured EmbeddingInferencePipeline

    Example::

        from histocoreml.config import FoundationConfig
        from histocoreml.pipelines.inference import create_embedding_pipeline

        cfg = FoundationConfig(model_name="uni")
        pipeline = create_embedding_pipeline(cfg)
        results = pipeline.run([Path("slide.svs")], output_dir=Path("embeddings"))
    """
    return EmbeddingInferencePipeline(cfg, encoder)
