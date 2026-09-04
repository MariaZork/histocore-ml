"""Inference pipelines for HistoCoreML.

Unified inference interface for all model types:
- Segmentation
- Feature extraction (embeddings)
- Classification (future)
"""

from histocoreml.pipelines.inference.embedding import (
    EmbeddingInferencePipeline,
    create_embedding_pipeline,
)
from histocoreml.pipelines.inference.segmentation import (
    SegmentationInferencePipeline,
    create_segmentation_pipeline,
)

__all__ = [
    "SegmentationInferencePipeline",
    "EmbeddingInferencePipeline",
    "create_segmentation_pipeline",
    "create_embedding_pipeline",
]
