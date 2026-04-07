"""TiffFile-backed WSI reader for generic TIFF / BigTIFF / OME-TIFF.

Does not require openslide. Suitable for standard tiled TIFFs produced by
common scanners when openslide support is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
        self._tif = None
        self._series = None

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

        levels = self._series.levels if self._series else [self._series]
        level_dimensions = tuple(
            (int(lv.shape[-1]), int(lv.shape[-2])) for lv in levels
        )
        base_w, base_h = level_dimensions[0]
        level_downsamples = tuple(
            base_w / w for w, h in level_dimensions
        )

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
        self._ensure_open()
        lv = self._series.levels[level]
        x, y = location
        w, h = size
        arr = lv.asarray()
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = arr.transpose(1, 2, 0)
        region = arr[y: y + h, x: x + w]
        if region.shape[2] == 4:
            region = region[:, :, :3]
        return region.astype(np.uint8)

    def get_thumbnail(self, max_size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
        self._ensure_open()
        meta = self.get_metadata()
        # Use the coarsest level as thumbnail
        thumb_level = meta.level_count - 1
        lv = self._series.levels[thumb_level]
        arr = lv.asarray()
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = arr.transpose(1, 2, 0)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._tif is None:
            raise RuntimeError(
                "Reader is not open. Use it as a context manager or call open() first."
            )

    def _extract_mpp(self) -> tuple[float | None, float | None]:
        """Try to extract MPP from OME-XML or TIFF resolution tags."""
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
