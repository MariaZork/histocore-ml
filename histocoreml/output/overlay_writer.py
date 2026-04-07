"""WSI overlay writer — blend the segmentation mask over a slide thumbnail."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from histocoreml.config import OutputConfig

if TYPE_CHECKING:
    from histocoreml.io.base_reader import BaseWSIReader

logger = logging.getLogger(__name__)

_OVERLAY_RGB: tuple[int, int, int] = (255, 0, 0)


def save_overlay(
    slide_path: Path,
    mask: np.ndarray,
    stem: str,
    cfg: OutputConfig,
    reader: BaseWSIReader,
) -> Path | None:
    """Blend the binary *mask* over a slide thumbnail and write a PNG.

    Args:
        slide_path: Path to the source WSI (used only for logging).
        mask: Binary uint8 array ``(H, W)`` with values 0 / 1.
        stem: Output filename stem.
        cfg: Output configuration.
        reader: An already-open :class:`BaseWSIReader` instance.

    Returns:
        Path to the saved PNG, or ``None`` on failure.
    """
    if not cfg.save_overlay:
        return None
    try:
        return _blend_and_save(slide_path, mask, stem, cfg, reader)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Overlay generation failed for %s — %s: %s. Mask output is unaffected.",
            slide_path.name,
            type(exc).__name__,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return None


def _blend_and_save(
    slide_path: Path,
    mask: np.ndarray,
    stem: str,
    cfg: OutputConfig,
    reader: BaseWSIReader,
) -> Path:
    from PIL import Image  # noqa: PLC0415

    max_edge = cfg.overlay_max_edge
    alpha = float(cfg.overlay_alpha)

    thumbnail = reader.get_thumbnail(max_size=(max_edge, max_edge))
    if thumbnail is None or thumbnail.size == 0:
        raise RuntimeError("Reader returned an empty thumbnail.")

    th_h, th_w = thumbnail.shape[:2]

    mask_resized = cv2.resize(
        (mask * 255).astype(np.uint8),
        (th_w, th_h),
        interpolation=cv2.INTER_NEAREST,
    )
    fg: NDArray[np.bool_] = mask_resized > 127

    overlay: NDArray[np.float32] = thumbnail.astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0

    blended = overlay.copy()
    blended[fg] = (1.0 - alpha) * overlay[fg] + alpha * red[fg]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"{stem}_overlay.png"
    Image.fromarray(blended).save(str(out_path))

    fg_pct = 100.0 * float(fg.mean())
    logger.info(
        "Overlay saved → %s (foreground=%.1f%%, alpha=%.2f, thumbnail=%d×%d)",
        out_path,
        fg_pct,
        alpha,
        th_w,
        th_h,
    )
    return out_path
