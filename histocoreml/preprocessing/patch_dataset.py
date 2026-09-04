"""PyTorch Dataset and DataLoader for WSI patch extraction.

:class:`PatchDataset` is a standard map-style ``torch.utils.data.Dataset``.
Each item is a ``{"image": Tensor, "coord": PatchCoord}`` dict (or ``None``
for background patches) loaded on demand by DataLoader workers.

Design notes
------------
* Map-style — the DataLoader distributes work across ``num_workers`` processes
  cleanly and supports ``prefetch_factor`` buffering.
* The tissue filter runs inside the worker; background patches are returned as
  ``None`` and dropped by the collator before they reach the model.
* Stain normalisation can be applied per-patch by passing ``normalise=True``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from histocoreml.config import TilingConfig
from histocoreml.io.base_reader import BaseWSIReader
from histocoreml.io.factory import get_reader
from histocoreml.preprocessing.patch_coord import PatchCoord
from histocoreml.preprocessing.patch_utils import (
    is_tissue as _is_tissue,
)
from histocoreml.preprocessing.patch_utils import (
    macenko_normalise as _macenko_normalise,
)
from histocoreml.preprocessing.patch_utils import (
    pad_to_size as _pad_to_size,
)
from histocoreml.preprocessing.patch_utils import (
    rescale_patch as _rescale_patch,
)

logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """Map-style PyTorch dataset over a WSI's patch coordinate list.

    Args:
        slide_path:       Path to the WSI file.
        coords:           Pre-computed list of :class:`PatchCoord` objects.
        tiling_cfg:       Tiling configuration for tissue filtering.
        model_patch_size: Target patch size (H = W) for the model.
        normalise:        If True, apply Macenko H&E normalisation per patch.
    """

    def __init__(
        self,
        slide_path: Path,
        coords: list[PatchCoord],
        tiling_cfg: TilingConfig,
        model_patch_size: int,
        normalise: bool = False,
    ) -> None:
        self._slide_path = slide_path
        self._reader: BaseWSIReader | None = None
        self._coords = coords
        self._tiling_cfg = tiling_cfg
        self._model_patch_size = model_patch_size
        self._normalise = normalise
        self._reader = None

    def _get_reader(self) -> BaseWSIReader:
        if self._reader is None:
            self._reader = get_reader(self._slide_path)
            self._reader.open()
        return self._reader

    def __del__(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self._coords)

    def __getitem__(self, idx: int) -> dict | None:
        coord = self._coords[idx]
        reader = self._get_reader()

        patch = reader.read_region(
            location=(coord.x, coord.y),
            level=coord.level,
            size=(coord.patch_size, coord.patch_size),
        )
        patch = _rescale_patch(patch, coord, self._model_patch_size)
        patch = _pad_to_size(patch, self._model_patch_size)

        if not _is_tissue(patch, self._tiling_cfg):
            return None

        if self._normalise:
            patch = _macenko_normalise(patch)

        tensor = torch.from_numpy(patch).permute(2, 0, 1).float().div(255.0)
        return {"image": tensor, "coord": coord}


def build_dataloader(
    slide_path: Path,
    coords: list[PatchCoord],
    tiling_cfg: TilingConfig,
    model_patch_size: int,
    batch_size: int,
    normalise: bool = False,
    transform: Callable[..., dict[str, np.ndarray]] | None = None,
) -> DataLoader:
    """Build a DataLoader for WSI patch inference.

    Args:
        slide_path:       Path to the WSI file.
        coords:           Pre-computed patch coordinate list.
        tiling_cfg:       Tiling configuration.
        model_patch_size: Target patch size.
        batch_size:       Patches per batch.
        normalise:        Apply Macenko normalisation per patch.

    Returns:
        Configured :class:`torch.utils.data.DataLoader`.
    """
    dataset = PatchDataset(slide_path, coords, tiling_cfg, model_patch_size, normalise)
    nw = tiling_cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate_skip_none,
        num_workers=nw,
        prefetch_factor=tiling_cfg.prefetch_factor if nw > 0 else None,
        persistent_workers=nw > 0,
        pin_memory=False,
        drop_last=False,
    )


def _collate_skip_none(batch: list[dict | None]) -> dict | None:
    """Collate a list of dataset items, silently dropping ``None`` entries."""
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return {
        "images": torch.stack([item["image"] for item in valid]),
        "coords": [item["coord"] for item in valid],
    }
