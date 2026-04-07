"""Abstract base class for all foundation model patch encoders."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch

from histocoreml.config import FoundationConfig


class BaseEncoder(abc.ABC):
    """Abstract interface for patch-level feature encoders.

    Usage::

        with MyEncoder(cfg) as enc:
            embeddings = enc.encode_batch(batch_tensor)  # (N, D)
    """

    def __init__(self, cfg: FoundationConfig) -> None:
        self._cfg = cfg
        self._device = torch.device(cfg.device)
        self._model: torch.nn.Module | None = None

    @property
    def embedding_dim(self) -> int:
        return self._cfg.embedding_dim

    @abc.abstractmethod
    def load(self) -> BaseEncoder:
        """Load model weights. Returns self."""

    def __enter__(self) -> BaseEncoder:
        return self.load()

    def __exit__(self, *_: object) -> None:
        pass

    @abc.abstractmethod
    @torch.inference_mode()
    def encode_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Encode a batch of patches.

        Args:
            batch: Float32 tensor (N, 3, H, W) normalised to [0, 1].

        Returns:
            float32 array (N, embedding_dim).
        """

    def encode_batch_normalised(self, batch: torch.Tensor) -> np.ndarray:
        """Encode and L2-normalise embeddings."""
        emb = self.encode_batch(batch)
        if self._cfg.normalize_embeddings:
            norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-8)
            emb = emb / norms
        return emb

    @abc.abstractmethod
    def get_transform(self) -> Any:
        """Return the torchvision transform expected by this encoder."""
