"""PatchCoord: immutable descriptor for a single patch location on a WSI pyramid."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchCoord:
    """Immutable descriptor of a single patch location on a WSI pyramid."""

    x: int
    """Top-left X in level-0 pixel space."""

    y: int
    """Top-left Y in level-0 pixel space."""

    level: int
    """Pyramid level at which the patch is read."""

    patch_size: int
    """Spatial dimension of the patch (H = W) in *level* pixel space."""

    col_idx: int
    """Column index of this patch in the tiling grid."""

    row_idx: int
    """Row index of this patch in the tiling grid."""

    rescale_factor: float = 1.0
    """If != 1.0, the read region must be rescaled to reach target_mpp."""

    slide_id: str = ""
    """Optional identifier for the source slide (useful in multi-slide batches)."""

    @property
    def grid_position(self):
        """(row_idx, col_idx) tuple."""
        return (self.row_idx, self.col_idx)
