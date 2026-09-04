"""HistoCoreML Pipelines — Unified inference and training workflows.

This module provides a unified interface for all ML pipelines in HistoCoreML,
including inference, training, and feature extraction.

Inference Pipelines::

    from histocoreml.pipelines import create_pipeline
    from histocoreml.config import SegmentationPipelineConfig

    # Segmentation inference
    cfg = SegmentationPipelineConfig.from_yaml("configs/default.yaml")
    pipeline = create_pipeline("segmentation", cfg)
    results = pipeline.run([Path("slide.svs")])

    # Foundation model inference (embeddings)
    from histocoreml.pipelines import create_embedding_pipeline
    cfg = FoundationConfig(model_name="uni")
    pipeline = create_embedding_pipeline(cfg)
    results = pipeline.run([Path("slide.svs")])

Training Pipelines::

    from histocoreml.config import ExperimentConfig
    from histocoreml.pipelines import SegmentationTrainingPipeline

    cfg = ExperimentConfig.from_yaml("configs/hubmap_glomeruli.yaml")
    pipeline = SegmentationTrainingPipeline(cfg)
    result = pipeline.run()  # Tiles slides on the fly → train → validate → infer

Base Classes::

    from histocoreml.pipelines.base import (
        BasePipeline,           # Generic base for all pipelines
        BaseInferencePipeline,  # Base for inference pipelines
        BaseTrainingPipeline,   # Base for training pipelines
        InferenceResult,        # Base result for inference
        TrainingResult,         # Base result for training
    )
"""

from typing import Any

from histocoreml.config import (
    InferenceResult,
    PipelineConfig,
    SegmentationInferenceResult,
    TrainingResult,
)
from histocoreml.pipelines.base import (
    BaseInferencePipeline,
    BasePipeline,
    BaseTrainingPipeline,
)
from histocoreml.pipelines.inference.embedding import (
    EmbeddingInferencePipeline,
    create_embedding_pipeline,
)

# Inference pipelines and factories
from histocoreml.pipelines.inference.segmentation import (
    SegmentationInferencePipeline,
    create_segmentation_pipeline,
)

# Training pipelines
from histocoreml.pipelines.training.segmentation import (
    ExtractionStats,
    SegmentationTrainingPipeline,
    extract_patches_to_disk,
)

__all__ = [
    # Base classes
    "BasePipeline",
    "BaseInferencePipeline",
    "BaseTrainingPipeline",
    "PipelineConfig",
    "InferenceResult",
    "SegmentationInferenceResult",
    "TrainingResult",
    # Inference pipelines
    "SegmentationInferencePipeline",
    "EmbeddingInferencePipeline",
    # Training pipelines
    "SegmentationTrainingPipeline",
    "extract_patches_to_disk",
    "ExtractionStats",
    # Factory functions
    "create_segmentation_pipeline",
    "create_embedding_pipeline",
]


def create_pipeline(pipeline_type: str, config: Any, **kwargs: Any) -> BaseInferencePipeline:
    """Factory function to create any pipeline by type.

    Args:
        pipeline_type: Type of pipeline ('segmentation', 'embedding', etc.)
        config: Pipeline configuration
        **kwargs: Additional arguments for specific pipeline types

    Returns:
        Pipeline instance

    Example::

        # Segmentation inference
        from histocoreml.config import SegmentationPipelineConfig
        cfg = SegmentationPipelineConfig.from_yaml("configs/default.yaml")
        pipeline = create_pipeline("segmentation", cfg)

        # Embedding inference
        from histocoreml.config import FoundationConfig
        cfg = FoundationConfig(model_name="uni")
        pipeline = create_pipeline("embedding", cfg)
    """
    if pipeline_type == "segmentation":
        return create_segmentation_pipeline(config)
    elif pipeline_type == "embedding":
        return create_embedding_pipeline(config, **kwargs)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
