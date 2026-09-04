"""TiffFile-backed WSI reader for generic TIFF / BigTIFF / OME-TIFF.

Does not require openslide. Suitable for standard tiled TIFFs produced by
common scanners when openslide support is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from histocoreml.io.base_reader import BaseWSIReader, WSIMetadata

logger = logging.getLogger(__name__)


class TifffileReader(BaseWSIReader):
    """WSI reader backed by the ``tifffile`` library.

    Reads pyramid TIFFs (tiled, multi-resolution) and OME-TIFF files.
    MPP is extracted from OME-XML or TIFF resolution tags when available.

    Usage::

        with TifffileReader(path) as reader:
            meta  = reader.get_metadata()
            patch = reader.read_region((x, y), level=0, size=(512, 512))
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._tif: Any | None = None
        self._series: Any | None = None

    def open(self) -> TifffileReader:
        try:
            import tifffile  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "tifffile is required for this backend: pip install tifffile"
            ) from exc

        logger.debug("Opening TIFF with tifffile: %s", self._path)
        self._tif = tifffile.TiffFile(str(self._path))
        self._series = self._tif.series[0] if self._tif.series else None
        return self

    def close(self) -> None:
        if self._tif is not None:
            self._tif.close()
            self._tif = None
            self._series = None
            logger.debug("Closed tifffile handle: %s", self._path)

    def get_metadata(self) -> WSIMetadata:
        self._ensure_open()
        assert self._series is not None

        levels = self._series.levels
        level_dimensions = tuple(self._level_size(lv) for lv in levels)
        base_w, _base_h = level_dimensions[0]
        level_downsamples = tuple(base_w / w for w, _ in level_dimensions)

        mpp_x, mpp_y = self._extract_mpp()

        return WSIMetadata(
            path=self._path,
            level_count=len(level_dimensions),
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            vendor=None,
            properties={},
        )

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Read a region and return an RGB uint8 array ``(h, w, 3)``.

        Always returns exactly the requested size, zero-padding whatever falls
        outside the level — the same contract OpenSlide provides, so the two
        backends stay interchangeable. Slicing the array directly would instead
        let a negative coordinate wrap around and silently yield an empty or
        wrong-edge patch.

        Args:
            location: Top-left ``(x, y)`` in *level* pixel space.
            level:    Pyramid level to read from.
            size:     ``(width, height)`` of the region.

        Returns:
            uint8 array of shape ``(height, width, 3)``.

        Raises:
            ValueError: If width or height is not positive.
        """
        self._ensure_open()
        assert self._series is not None

        x, y = location
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError(f"read_region size must be positive, got {size}")

        arr = self._level_array(level)
        level_h, level_w = arr.shape[:2]

        region: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)

        # Clip the requested window to what the level actually holds.
        src_x0, src_y0 = max(x, 0), max(y, 0)
        src_x1, src_y1 = min(x + w, level_w), min(y + h, level_h)
        if src_x1 > src_x0 and src_y1 > src_y0:
            region[src_y0 - y : src_y1 - y, src_x0 - x : src_x1 - x] = arr[
                src_y0:src_y1, src_x0:src_x1
            ]

        return region

    def get_thumbnail(self, max_size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
        self._ensure_open()
        assert self._series is not None
        # Use the coarsest level as thumbnail
        thumb_level = len(self._series.levels) - 1
        return self._level_array(thumb_level)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _level_size(level: Any) -> tuple[int, int]:
        """Return ``(width, height)`` for a pyramid level.

        TIFF stores pixels channel-last (``YXS``) or channel-first (``SYX``),
        and grayscale pages carry no channel axis at all. Reading the ``axes``
        string is what keeps a (H, W, 3) page from being measured as 3 px wide.
        """
        axes = getattr(level, "axes", "")
        shape = tuple(int(dim) for dim in level.shape)

        if "Y" in axes and "X" in axes:
            return shape[axes.index("X")], shape[axes.index("Y")]

        # No axes metadata: assume channel-first only when the leading axis
        # looks like a channel count.
        if len(shape) == 3 and shape[0] in (3, 4):
            return shape[2], shape[1]
        return shape[1], shape[0]

    def _level_array(self, level: int) -> np.ndarray:
        """Return a level as a channel-last RGB uint8 array ``(H, W, 3)``."""
        assert self._series is not None
        lv = self._series.levels[level]
        arr = lv.asarray()

        axes = getattr(lv, "axes", "")
        if "S" in axes and axes.index("S") == 0:
            arr = np.moveaxis(arr, 0, -1)
        elif not axes and arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = arr.transpose(1, 2, 0)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return arr.astype(np.uint8, copy=False)

    def _ensure_open(self) -> None:
        if self._tif is None:
            raise RuntimeError(
                "Reader is not open. Use it as a context manager or call open() first."
            )

    def _extract_mpp(self) -> tuple[float | None, float | None]:
        """Try to extract MPP from OME-XML or TIFF resolution tags."""
        assert self._tif is not None
        try:
            if self._tif.ome_metadata:
                import xml.etree.ElementTree as ET  # noqa: PLC0415

                root = ET.fromstring(self._tif.ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                px = root.find(".//ome:Pixels", ns)
                if px is not None:
                    sx = float(px.get("PhysicalSizeX", 0))
                    sy = float(px.get("PhysicalSizeY", 0))
                    unit = px.get("PhysicalSizeXUnit", "µm")
                    if unit in ("µm", "um", "micrometer"):
                        return sx or None, sy or None
        except (AttributeError, ValueError, TypeError):
            pass

        # Fall back to TIFF resolution tags
        try:
            page = self._tif.pages[0]
            xres = page.tags.get("XResolution")
            yres = page.tags.get("YResolution")
            unit = page.tags.get("ResolutionUnit")
            if xres and yres and unit:
                unit_val = unit.value if hasattr(unit, "value") else unit
                xr = xres.value
                yr = yres.value
                if isinstance(xr, tuple):
                    xr = xr[0] / xr[1]
                    yr = yr[0] / yr[1]
                if unit_val == 3:  # centimeters
                    return 1e4 / xr, 1e4 / yr
                if unit_val == 2:  # inches
                    return 25400 / xr, 25400 / yr
        except (AttributeError, ValueError, TypeError, ZeroDivisionError):
            pass

        return None, None
