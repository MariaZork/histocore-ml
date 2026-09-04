"""Abstract base for Whole Slide Image readers.

New file-format backends (e.g. DICOM, JP2K, TiffFile) can be added by
subclassing ``BaseWSIReader`` without touching any other pipeline component.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WSIMetadata:
    """Immutable metadata extracted from a WSI file."""

    path: Path
    """Absolute path to the slide file."""

    level_count: int
    """Total number of resolution levels in the pyramid."""

    level_dimensions: tuple[tuple[int, int], ...]
    """(width, height) for every pyramid level, ordered fine → coarse."""

    level_downsamples: tuple[float, ...]
    """Downsample factor relative to level-0 for each pyramid level."""

    mpp_x: float | None
    """Microns-per-pixel in the X direction at level 0 (None if unavailable)."""

    mpp_y: float | None
    """Microns-per-pixel in the Y direction at level 0 (None if unavailable)."""

    vendor: str | None
    """Scanner vendor string from slide metadata (None if unavailable)."""

    properties: dict[str, str] = None  # type: ignore[assignment]
    """Raw key-value properties from the slide (scanner-specific)."""

    @property
    def dimensions(self) -> tuple[int, int]:
        """Width × height of the highest-resolution level (level 0)."""
        return self.level_dimensions[0]

    @property
    def mpp(self) -> float | None:
        """Isotropic mpp estimate. Falls back to whichever axis is available."""
        if self.mpp_x is not None and self.mpp_y is not None:
            return (self.mpp_x + self.mpp_y) / 2.0
        return self.mpp_x or self.mpp_y

    def best_level_for_mpp(self, target_mpp: float) -> tuple[int, float]:
        """Find the pyramid level closest to *target_mpp*.

        Args:
            target_mpp: Desired resolution in microns-per-pixel.

        Returns:
            Tuple of (level_index, actual_mpp_at_level).

        Raises:
            ValueError: If slide mpp is unknown.
        """
        if self.mpp is None:
            raise ValueError(f"Cannot compute best level: mpp metadata missing for {self.path}")
        target_downsample = target_mpp / self.mpp
        best_level = 0
        best_diff = abs(self.level_downsamples[0] - target_downsample)
        for lvl, ds in enumerate(self.level_downsamples):
            diff = abs(ds - target_downsample)
            if diff < best_diff:
                best_diff = diff
                best_level = lvl
        actual_mpp = self.mpp * self.level_downsamples[best_level]
        return best_level, actual_mpp

    def level_for_mpp(self, target_mpp: float) -> tuple[int, float]:
        """Resolve the pyramid level for *target_mpp*, tolerating missing metadata.

        Same as :meth:`best_level_for_mpp`, except that a slide with no MPP
        recorded (common for public TIFF datasets) resolves to level 0 at the
        requested resolution instead of raising. Tiling and mask assembly must
        agree on the level, so both go through this.

        Args:
            target_mpp: Desired resolution in microns-per-pixel.

        Returns:
            Tuple of (level_index, actual_mpp_at_level).
        """
        try:
            return self.best_level_for_mpp(target_mpp)
        except ValueError:
            logger.warning(
                "MPP metadata missing for %s — using level 0 at target_mpp=%.4f",
                self.path,
                target_mpp,
            )
            return 0, target_mpp

    def magnification_at_level(self, level: int, objective_power: float = 40.0) -> float:
        """Estimate effective magnification at a pyramid level.

        Args:
            level: Pyramid level index.
            objective_power: Nominal objective magnification (default 40×).

        Returns:
            Estimated magnification as a float.
        """
        return objective_power / self.level_downsamples[level]


class BaseWSIReader(abc.ABC):
    """Abstract interface for reading Whole Slide Images.

    Concrete implementations:
    - :class:`~histocoreml.io.openslide_reader.OpenSlideReader`
    - :class:`~histocoreml.io.tifffile_reader.TifffileReader`
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"WSI not found: {self._path}")

    @property
    def path(self) -> Path:
        return self._path

    @abc.abstractmethod
    def open(self) -> BaseWSIReader:
        """Open the slide file and prepare internal state."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all file handles and internal resources."""

    def __enter__(self) -> BaseWSIReader:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    @abc.abstractmethod
    def get_metadata(self) -> WSIMetadata:
        """Return slide metadata without reading pixel data."""

    @abc.abstractmethod
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Read a rectangular region from the slide pyramid.

        Args:
            location: (x, y) top-left corner in *level-0* pixel coordinates.
            level: Pyramid level to read from (0 = highest resolution).
            size: (width, height) of the region in *level* pixel coordinates.

        Returns:
            NumPy array of shape (height, width, channels) with dtype uint8.
        """

    @abc.abstractmethod
    def get_thumbnail(self, max_size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """Return a downsampled RGB thumbnail of the entire slide.

        Args:
            max_size: Maximum (width, height) of the thumbnail.

        Returns:
            NumPy array of shape (H, W, 3) dtype uint8.
        """

    def read_region_at_mpp(
        self,
        location: tuple[int, int],
        target_mpp: float,
        size_um: tuple[float, float],
    ) -> np.ndarray:
        """Read a region specified in physical (micron) coordinates.

        Args:
            location: (x, y) top-left corner in level-0 pixels.
            target_mpp: Desired resolution in µm/px.
            size_um: (width_µm, height_µm) of the region in microns.

        Returns:
            NumPy array of shape (H, W, 3) uint8 at the requested resolution.
        """
        metadata = self.get_metadata()
        level, actual_mpp = metadata.best_level_for_mpp(target_mpp)
        w_px = max(1, round(size_um[0] / actual_mpp))
        h_px = max(1, round(size_um[1] / actual_mpp))
        return self.read_region(location, level, (w_px, h_px))
