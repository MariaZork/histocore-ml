"""Grid generator: computes patch coordinates across the WSI pyramid.

MPP enforcement strategy
------------------------
The model expects patches at **exactly** ``model_cfg.target_mpp`` (e.g., 0.88 µm/px).

``best_level_for_mpp`` selects the pyramid level whose MPP is *closest* to the
target. The ratio ``actual_mpp / target_mpp`` is stored as ``rescale_factor``
on each :class:`~histocoreml.preprocessing.patch_coord.PatchCoord`.

When ``rescale_factor != 1.0``, patch readers call
:func:`~histocoreml.preprocessing.patch_utils.rescale_patch` to resize the
extracted region to the canonical ``patch_size × patch_size`` at target_mpp.
"""

from __future__ import annotations

import logging

from histocoreml.config import ModelConfig, TilingConfig
from histocoreml.io.base_reader import WSIMetadata
from histocoreml.preprocessing.patch_coord import PatchCoord

logger = logging.getLogger(__name__)


def generate_patch_coords(
    metadata: WSIMetadata,
    model_cfg: ModelConfig,
    tiling_cfg: TilingConfig,
    slide_id: str = "",
) -> list[PatchCoord]:
    """Compute all patch coordinates for a slide at the target resolution.

    Each :class:`PatchCoord` describes:

    * ``(x, y)``         — top-left corner in **level-0** pixel space.
    * ``level``          — pyramid level used for reading.
    * ``patch_size``     — pixels to *request* from the reader at ``level``.
    * ``rescale_factor`` — ``actual_level_mpp / target_mpp``.

    Args:
        metadata:   Slide metadata (dimensions, mpp, level info).
        model_cfg:  Model configuration (patch_size, target_mpp).
        tiling_cfg: Tiling configuration (overlap).
        slide_id:   Optional slide identifier attached to each coord.

    Returns:
        Row-major ordered list of :class:`PatchCoord` objects.

    Raises:
        ValueError: If overlap >= patch_size.
    """
    if tiling_cfg.overlap >= model_cfg.patch_size:
        raise ValueError(
            f"patch_size ({model_cfg.patch_size}) must be larger than "
            f"overlap ({tiling_cfg.overlap})"
        )

    level, actual_mpp = metadata.best_level_for_mpp(model_cfg.target_mpp)
    rescale: float = actual_mpp / model_cfg.target_mpp

    logger.info(
        "MPP enforcement | target=%.4f µm/px | selected level=%d "
        "(actual=%.4f µm/px) | rescale_factor=%.4f%s",
        model_cfg.target_mpp,
        level,
        actual_mpp,
        rescale,
        "" if abs(rescale - 1.0) < 0.01 else " ← patch resize will be applied",
    )

    read_size: int = max(1, round(model_cfg.patch_size * rescale))
    target_stride = model_cfg.patch_size - tiling_cfg.overlap
    level_stride: int = max(1, round(target_stride * rescale))

    w, h = metadata.level_dimensions[level]
    ds = metadata.level_downsamples[level]

    coords: list[PatchCoord] = []
    row_idx = 0
    y = 0
    while y < h:
        col_idx = 0
        x = 0
        while x < w:
            coords.append(
                PatchCoord(
                    x=int(x * ds),
                    y=int(y * ds),
                    level=level,
                    patch_size=read_size,
                    col_idx=col_idx,
                    row_idx=row_idx,
                    rescale_factor=rescale,
                    slide_id=slide_id,
                )
            )
            x += level_stride
            col_idx += 1
        y += level_stride
        row_idx += 1

    logger.info(
        "Generated %d patch coords (read_size=%d px at level %d, "
        "model input=%d px after rescale).",
        len(coords), read_size, level, model_cfg.patch_size,
    )
    return coords
