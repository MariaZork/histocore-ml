"""MemmapCanvas: memory-mapped accumulator for large WSI mask assembly."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class MemmapCanvas:
    """Two memory-mapped uint32 arrays: an accumulator and a hit-count map.

    Both arrays are stored in a temporary directory so that the OS can
    page them in/out — the full canvas never occupies RAM regardless of
    slide dimensions.

    Attributes:
        accumulator: Running sum of per-pixel predictions.
        counts:      Number of patches that contributed to each pixel.
        shape:       ``(height, width)`` of the canvas.
        tmpdir:      Path to the temporary directory holding the mmap files.
    """

    accumulator: np.ndarray
    counts: np.ndarray
    shape: Tuple[int, int]
    tmpdir: Path

    @classmethod
    def create(cls, height: int, width: int) -> "MemmapCanvas":
        """Allocate a fresh canvas backed by temp files.

        Args:
            height: Canvas height in pixels.
            width:  Canvas width in pixels.

        Returns:
            Initialised :class:`MemmapCanvas` with all zeros.
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="histocoreml_canvas_"))
        shape = (height, width)
        return cls(
            accumulator=np.memmap(
                str(tmpdir / "accumulator.dat"), dtype=np.uint32, mode="w+", shape=shape
            ),
            counts=np.memmap(
                str(tmpdir / "counts.dat"), dtype=np.uint32, mode="w+", shape=shape
            ),
            shape=shape,
            tmpdir=tmpdir,
        )

    def finalise(self, threshold: float = 0.5) -> np.ndarray:
        """Average the accumulated predictions and return a binary mask.

        Args:
            threshold: Binarisation threshold on the averaged probabilities.

        Returns:
            uint8 array of shape ``(H, W)`` with values 0 or 1.
        """
        safe_counts = np.where(self.counts > 0, self.counts, 1)
        avg = self.accumulator.astype(np.float32) / safe_counts
        return (avg >= threshold).astype(np.uint8)

    def finalise_proba(self) -> np.ndarray:
        """Return the averaged probability map as float32 (H, W) in [0, 1]."""
        safe_counts = np.where(self.counts > 0, self.counts, 1)
        return self.accumulator.astype(np.float32) / safe_counts

    def cleanup(self) -> None:
        """Delete the temporary directory and all mmap files."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
