"""Concrete mask writers: TIFF, NumPy, Zarr, GeoJSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from histocoreml.config import OutputConfig
from histocoreml.io.base_reader import WSIMetadata
from histocoreml.output.base_writer import BaseMaskWriter, WriteResult

logger = logging.getLogger(__name__)


# ── TIFF ──────────────────────────────────────────────────────────────────────

class TiffMaskWriter(BaseMaskWriter):
    """Write the mask as a tiled, spatially-referenced GeoTIFF.

    The output TIFF carries the slide's mpp as a resolution tag so that
    downstream tools (QuPath, ASAP, ImageJ) can overlay it correctly.
    """

    def write(self, mask: np.ndarray, metadata: WSIMetadata, stem: str) -> WriteResult:
        try:
            import tifffile  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("tifffile is required: pip install tifffile") from exc

        out_path = self._cfg.output_dir / f"{stem}_mask.tiff"
        compression = self._resolve_compression(self._cfg.compression)
        resolution, resolutionunit = self._build_resolution_tag(metadata)

        logger.info("Writing TIFF mask → %s", out_path)
        tifffile.imwrite(
            str(out_path),
            (mask * 255).astype(np.uint8),
            compression=compression,
            tile=(512, 512),
            resolution=resolution,
            resolutionunit=resolutionunit,
            photometric="minisblack",
        )

        thumbnail_path: Optional[Path] = None
        if self._cfg.save_thumbnail:
            thumbnail_path = self._write_thumbnail(mask, stem)

        return WriteResult(path=out_path, shape=mask.shape[:2], format="tiff",
                           thumbnail_path=thumbnail_path)

    def _write_thumbnail(self, mask: np.ndarray, stem: str) -> Optional[Path]:
        try:
            import cv2  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            logger.warning("opencv / Pillow not available; skipping thumbnail.")
            return None
        thumb_path = self._cfg.output_dir / f"{stem}_thumbnail.png"
        small = cv2.resize(mask.astype(np.uint8) * 255, (512, 512),
                           interpolation=cv2.INTER_NEAREST)
        Image.fromarray(small).save(str(thumb_path))
        return thumb_path

    @staticmethod
    def _build_resolution_tag(metadata: WSIMetadata):
        if metadata.mpp is None:
            return None, None
        ppcm = 1e4 / metadata.mpp
        return (ppcm, ppcm), 3

    @staticmethod
    def _resolve_compression(compression: str) -> str:
        if compression != "lzw":
            return compression
        try:
            import imagecodecs  # noqa: F401,PLC0415
            return compression
        except ImportError:
            logger.warning("imagecodecs not available; falling back to deflate.")
            return "deflate"


# ── NumPy ─────────────────────────────────────────────────────────────────────

class NumpyMaskWriter(BaseMaskWriter):
    """Write the mask as a NumPy ``.npy`` binary file."""

    def write(self, mask: np.ndarray, metadata: WSIMetadata, stem: str) -> WriteResult:
        out_path = self._cfg.output_dir / f"{stem}_mask.npy"
        logger.info("Writing NumPy mask → %s", out_path)
        np.save(str(out_path), mask)
        return WriteResult(path=out_path, shape=mask.shape[:2], format="npy")


# ── Zarr ──────────────────────────────────────────────────────────────────────

class ZarrMaskWriter(BaseMaskWriter):
    """Write the mask as a chunked Zarr array.

    Zarr supports cloud storage backends (S3, GCS, Azure) and scales to
    multi-terabyte whole-slide masks via lazy chunked access.

    Usage::

        cfg    = OutputConfig(output_format="zarr", output_dir=Path("outputs"))
        writer = ZarrMaskWriter(cfg)
        result = writer.write(mask, metadata, stem="slide_001")
        # → outputs/slide_001_mask.zarr
    """

    def write(self, mask: np.ndarray, metadata: WSIMetadata, stem: str) -> WriteResult:
        try:
            import zarr  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("zarr is required: pip install zarr") from exc

        out_path = self._cfg.output_dir / f"{stem}_mask.zarr"
        logger.info("Writing Zarr mask → %s", out_path)

        store = zarr.DirectoryStore(str(out_path))
        z = zarr.open(store, mode="w", shape=mask.shape, dtype=np.uint8,
                      chunks=(512, 512), compressor=zarr.Blosc(cname="lz4", clevel=5))
        z[:] = mask

        # Attach spatial metadata as Zarr attributes
        z.attrs["mpp"] = metadata.mpp
        z.attrs["slide_path"] = str(metadata.path)
        z.attrs["shape"] = list(mask.shape)

        return WriteResult(path=out_path, shape=mask.shape[:2], format="zarr")


# ── GeoJSON ───────────────────────────────────────────────────────────────────

class GeoJSONMaskWriter(BaseMaskWriter):
    """Export segmentation contours as GeoJSON polygons.

    Converts the binary mask to vector contours using OpenCV, then writes
    a GeoJSON FeatureCollection. Compatible with QuPath, ASAP, and web maps.

    Usage::

        cfg    = OutputConfig(output_format="geojson")
        writer = GeoJSONMaskWriter(cfg)
        result = writer.write(mask, metadata, stem="slide_001")
        # → outputs/slide_001_mask.geojson
    """

    def write(self, mask: np.ndarray, metadata: WSIMetadata, stem: str) -> WriteResult:
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("opencv-python is required: pip install opencv-python") from exc

        out_path = self._cfg.output_dir / f"{stem}_mask.geojson"
        logger.info("Writing GeoJSON mask → %s", out_path)

        mpp = metadata.mpp or 1.0
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        features = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            # Scale pixels → microns
            coords_um = (contour.squeeze() * mpp).tolist()
            if not isinstance(coords_um[0], list):
                coords_um = [coords_um]
            coords_um.append(coords_um[0])  # close ring
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords_um]},
                "properties": {"classification": "tumor", "mpp": mpp},
            })

        geojson = {"type": "FeatureCollection", "features": features}
        with out_path.open("w") as fh:
            json.dump(geojson, fh)

        logger.info("GeoJSON: %d contour(s) exported → %s", len(features), out_path)
        return WriteResult(
            path=out_path, shape=mask.shape[:2], format="geojson",
            metadata={"num_contours": len(features)},
        )
