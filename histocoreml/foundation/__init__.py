"""HistoCoreML foundation models — patch encoders for feature extraction.

Supported encoders
------------------
``UNIEncoder``     — UNI (universal pathology encoder from MedIBL / HMS)
``CONCHEncoder``   — CONCH (contrastive language-image pretraining, AIMIL)
``PLIPEncoder``    — PLIP (pathology language-image pretraining, Zhu et al.)
``ViTEncoder``     — Generic ViT encoder via timm (custom weights)

Usage::

    from histocoreml.foundation import get_encoder
    from histocoreml.config import FoundationConfig
    from pathlib import Path

    cfg     = FoundationConfig(model_name="uni", target_mpp=0.5, batch_size=64)
    encoder = get_encoder(cfg)

For full embedding pipeline, use::

    from histocoreml.pipelines import EmbeddingInferencePipeline, create_embedding_pipeline
"""

from histocoreml.foundation.base_encoder import BaseEncoder
from histocoreml.foundation.factory import get_encoder
from histocoreml.foundation.vit_encoder import ViTEncoder

__all__ = [
    "BaseEncoder",
    "ViTEncoder",
    "get_encoder",
]
