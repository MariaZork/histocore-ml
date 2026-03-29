"""WriteResult dataclass and BaseMaskWriter abstract base class."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from histocoreml.config import OutputConfig
from histocoreml.io.base_reader import WSIMetadata


@dataclass(frozen=True)
class WriteResult:
    """Metadata describing a successfully written mask file."""

    path: Path
    """Absolute path to the written file."""

    shape: Tuple[int, int]
    """(height, width) of the mask in pixels."""

    format: str
    """File format string, e.g. ``'tiff'``, ``'npy'``, ``'rle_plain'``."""

    thumbnail_path: Optional[Path] = None
    """Optional path to a QC thumbnail overlay (PNG)."""

    metadata: Optional[dict] = None
    """Optional dict of extra metadata (compression ratio, run count, etc.)."""


class BaseMaskWriter(abc.ABC):
    """Abstract interface for persisting segmentation masks.

    Add new output formats (HDF5, Zarr, GeoJSON) by subclassing this and
    registering the new class in :func:`~histocoreml.output.factory.get_writer`.
    """

    def __init__(self, cfg: OutputConfig) -> None:
        self._cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def write(
        self,
        mask: np.ndarray,
        metadata: WSIMetadata,
        stem: str,
    ) -> WriteResult:
        """Write *mask* to disk and return a :class:`WriteResult`.

        Args:
            mask:     Binary uint8 array ``(H, W)`` with values 0 or 1.
            metadata: Slide metadata (used for spatial referencing tags).
            stem:     Output filename stem.
        """
