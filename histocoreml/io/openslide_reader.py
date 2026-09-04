"""OpenSlide-backed WSI reader.

Handles SVS, TIFF, NDPI, SCN, MRXS, and other formats supported by the
OpenSlide library.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from histocoreml.io.base_reader import BaseWSIReader, WSIMetadata

logger = logging.getLogger(__name__)


class OpenSlideReader(BaseWSIReader):
    """WSI reader backed by the ``openslide-python`` library.

    Usage::

        with OpenSlideReader(path) as reader:
            meta = reader.get_metadata()
            patch = reader.read_region((x, y), level=0, size=(512, 512))
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._slide: Any | None = None  # lazy-opened in open()

    def open(self) -> OpenSlideReader:
        try:
            import openslide  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openslide-python is required for SVS/NDPI/MRXS files.\n"
                "Install it via: pip install histocoreml[openslide]"
            ) from exc

        logger.debug("Opening WSI with OpenSlide: %s", self._path)
        self._slide = openslide.OpenSlide(str(self._path))
        return self

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
            self._slide = None
        logger.debug("Closed OpenSlide handle: %s", self._path)

    def get_metadata(self) -> WSIMetadata:
        self._ensure_open()
        assert self._slide is not None
        props = dict(self._slide.properties)

        mpp_x = self._parse_float(props.get("openslide.mpp-x"))
        mpp_y = self._parse_float(props.get("openslide.mpp-y"))

        metadata = WSIMetadata(
            path=self._path,
            level_count=self._slide.level_count,
            level_dimensions=tuple(tuple(d) for d in self._slide.level_dimensions),
            level_downsamples=tuple(self._slide.level_downsamples),
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            vendor=props.get("openslide.vendor"),
            properties=props,
        )

        w, h = metadata.dimensions
        mpp_str = f"{metadata.mpp:.4f} µm/px" if metadata.mpp else "unknown"
        ds_str = ", ".join(f"{d:.2f}" for d in metadata.level_downsamples)

        logger.info("Slide metadata | %s", self._path.name)
        logger.info("  Dimensions : %d × %d px (level 0)", w, h)
        logger.info("  Levels     : %d (downsamples: %s)", metadata.level_count, ds_str)
        logger.info(
            "  Resolution : %s (mpp_x=%s mpp_y=%s)",
            mpp_str,
            f"{mpp_x:.4f}" if mpp_x else "n/a",
            f"{mpp_y:.4f}" if mpp_y else "n/a",
        )
        logger.info("  Vendor     : %s", metadata.vendor or "unknown")

        if metadata.mpp is None:
            logger.warning(
                "Slide %s has no MPP metadata — target_mpp enforcement won't be applied.",
                self._path.name,
            )
        return metadata

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Read a region and return an RGB uint8 array (H, W, 3)."""
        self._ensure_open()
        assert self._slide is not None
        pil_img = self._slide.read_region(location, level, size)
        return np.array(pil_img.convert("RGB"), dtype=np.uint8)

    def get_thumbnail(self, max_size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
        self._ensure_open()
        assert self._slide is not None
        pil_thumb = self._slide.get_thumbnail(max_size)
        return np.array(pil_thumb.convert("RGB"), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._slide is None:
            raise RuntimeError(
                "Reader is not open. Use it as a context manager or call open() first."
            )

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None
