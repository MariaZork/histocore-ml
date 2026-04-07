"""MaskAssembler: incrementally assembles a full-slide mask from patch predictions."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from histocoreml.config import ModelConfig, TilingConfig
from histocoreml.io.base_reader import WSIMetadata
from histocoreml.postprocessing.memmap_canvas import MemmapCanvas
from histocoreml.preprocessing.patch_coord import PatchCoord

logger = logging.getLogger(__name__)


class MaskAssembler:
    """Incrementally assembles a full-slide mask from batches of patch predictions.

    Uses a :class:`~histocoreml.postprocessing.memmap_canvas.MemmapCanvas` so
    the canvas never fully occupies RAM. Overlapping patches are resolved by
    averaging (each pixel is divided by the number of patches that covered it).

    Usage::

        assembler = MaskAssembler(metadata, model_cfg, tiling_cfg)
        for masks, coords in inference_loop:
            assembler.add_batch(masks, coords)
        final_mask = assembler.finalise()
        assembler.cleanup()
    """

    def __init__(
        self,
        metadata: WSIMetadata,
        model_cfg: ModelConfig,
        tiling_cfg: TilingConfig,
        downsample_factor: int = 1,
    ) -> None:
        self._metadata = metadata
        self._downsample_factor = downsample_factor

        inf_level, _ = metadata.best_level_for_mpp(model_cfg.target_mpp)
        self._inf_ds = metadata.level_downsamples[inf_level]

        inf_w, inf_h = metadata.level_dimensions[inf_level]
        canvas_h = max(1, inf_h // downsample_factor)
        canvas_w = max(1, inf_w // downsample_factor)

        self._canvas = MemmapCanvas.create(canvas_h, canvas_w)
        logger.info(
            "MaskAssembler canvas: (%d × %d px) at inference level %d "
            "(ds=%.1f) backed by memmap in %s",
            canvas_w,
            canvas_h,
            inf_level,
            self._inf_ds,
            self._canvas.tmpdir,
        )

    def add_batch(self, masks: np.ndarray, coords: list[PatchCoord]) -> None:
        """Write a batch of patch predictions into the canvas.

        Args:
            masks: Binary uint8 array ``(N, H, W)``.
            coords: Corresponding :class:`PatchCoord` objects (same order).
        """
        for mask, coord in zip(masks, coords):
            self._write_patch(mask, coord)

    def add_proba_batch(self, probas: np.ndarray, coords: list[PatchCoord]) -> None:
        """Write soft probability predictions into the canvas.

        Args:
            probas: float32 array ``(N, H, W)`` with values in [0, 1].
            coords: Corresponding :class:`PatchCoord` objects.
        """
        # Scale to uint32 range for integer accumulation
        scaled: NDArray[np.uint32] = (probas * 65535).astype(np.uint32)
        for idx, coord in enumerate(coords):
            self._write_patch(scaled[idx], coord)

    def finalise(self) -> np.ndarray:
        """Average all accumulated predictions and return a binary mask.

        Returns:
            uint8 array ``(canvas_H, canvas_W)`` with values 0 or 1.
        """
        return self._canvas.finalise()

    def finalise_proba(self) -> np.ndarray:
        """Return averaged probability map as float32 (H, W)."""
        raw = self._canvas.finalise_proba()
        # If add_proba_batch was used, un-scale
        if raw.max() > 1.0:
            return raw / 65535.0
        return raw

    def cleanup(self) -> None:
        """Remove temporary memmap files from disk."""
        self._canvas.cleanup()
        logger.debug("Assembler canvas cleaned up.")

    def _write_patch(self, mask: np.ndarray, coord: PatchCoord) -> None:
        """Map a single patch mask into the correct canvas region."""
        canvas_h, canvas_w = self._canvas.shape

        cx = int(coord.x / self._inf_ds / self._downsample_factor)
        cy = int(coord.y / self._inf_ds / self._downsample_factor)
        ph, pw = mask.shape[0], mask.shape[1]

        y0 = max(cy, 0)
        x0 = max(cx, 0)
        y1 = min(cy + ph, canvas_h)
        x1 = min(cx + pw, canvas_w)

        if y1 <= y0 or x1 <= x0:
            return

        my0 = y0 - cy
        mx0 = x0 - cx
        my1 = my0 + (y1 - y0)
        mx1 = mx0 + (x1 - x0)

        self._canvas.accumulator[y0:y1, x0:x1] += mask[
            my0:my1, mx0:mx1
        ].astype(np.uint32)
        self._canvas.counts[y0:y1, x0:x1] += 1
