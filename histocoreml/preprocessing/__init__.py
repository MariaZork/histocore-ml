"""HistoCoreML preprocessing — patch extraction and tissue filtering."""

from histocoreml.preprocessing.grid_generator import (
    generate_patch_coords,
    generate_patch_grid,
)
from histocoreml.preprocessing.patch_coord import PatchCoord
from histocoreml.preprocessing.patch_dataset import PatchDataset, build_dataloader
from histocoreml.preprocessing.patch_utils import (
    ensure_rgb,
    filter_coords_by_tissue,
    is_tissue,
    pad_to_size,
    rescale_patch,
)

__all__ = [
    "PatchCoord",
    "generate_patch_coords",
    "generate_patch_grid",
    "rescale_patch",
    "is_tissue",
    "ensure_rgb",
    "filter_coords_by_tissue",
    "pad_to_size",
    "PatchDataset",
    "build_dataloader",
]
