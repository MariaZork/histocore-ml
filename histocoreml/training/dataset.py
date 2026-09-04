"""Datasets for segmentation training.

Two ways to feed a trainer, sharing this module because they are alternatives
for the same job:

:class:`SegmentationDataset`
    Tiles slides lazily. Patch coordinates are indexed once up front and pixels
    are read from the slide pyramid inside the DataLoader workers, so nothing
    intermediate is written to disk. This is the default: it needs no
    preparation step and adapts when tiling settings change.

:class:`PatchDirectoryDataset`
    Reads ``(image, mask)`` PNG pairs already on disk, as written by
    :func:`~histocoreml.pipelines.training.segmentation.extract_patches_to_disk`.
    Worth the extra pass when many experiments run over one dataset.

Ground truth for :class:`SegmentationDataset` comes from a
:class:`MaskProvider`, so the same dataset serves competition RLE strings,
GeoJSON annotations, or pre-rendered mask files by swapping the provider —
nothing here is specific to one dataset or organ.

Usage::

    from histocoreml.config import TilingConfig
    from histocoreml.training import RLEMaskProvider, SegmentationDataset

    dataset = SegmentationDataset(
        slide_dir=Path("data/train"),
        mask_provider=RLEMaskProvider.from_csv(Path("train.csv")),
        tiling_cfg=TilingConfig(overlap=256),
        patch_size=512,
        target_mpp=0.5,
    )
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from histocoreml.config import TilingConfig
from histocoreml.io.base_reader import BaseWSIReader
from histocoreml.io.factory import get_reader
from histocoreml.output.rle_codec import rle_decode
from histocoreml.preprocessing.grid_generator import generate_patch_grid
from histocoreml.preprocessing.patch_coord import PatchCoord
from histocoreml.preprocessing.patch_utils import (
    ensure_rgb,
    filter_coords_by_tissue,
    pad_to_size,
    rescale_patch,
)
from histocoreml.utils.progress import create_progress_bar

logger = logging.getLogger(__name__)

_DEFAULT_SLIDE_EXTENSIONS: tuple[str, ...] = (".tiff", ".tif", ".svs", ".ndpi", ".mrxs")
_TILE_EXTENSIONS = {".png", ".jpg", ".tif", ".tiff"}


# ── Mask providers ────────────────────────────────────────────────────────────


class MaskProvider(abc.ABC):
    """Supplies the full-resolution ground-truth mask for a slide."""

    @abc.abstractmethod
    def get_mask(self, slide_id: str, shape: tuple[int, int]) -> NDArray[np.uint8]:
        """Return the binary mask ``(H, W)`` for *slide_id* at *shape*."""

    @abc.abstractmethod
    def slide_ids(self) -> list[str]:
        """Return every slide id this provider can supply a mask for."""


class RLEMaskProvider(MaskProvider):
    """Masks decoded from run-length encoded strings (the Kaggle/HuBMAP format).

    Decoding a slide-sized mask is expensive, so the most recently used masks
    are cached. Patches from one slide are indexed contiguously, so a small
    cache is enough to make the common access pattern a hit.

    Args:
        encodings:  Mapping of ``slide_id`` → RLE string.
        cache_size: Number of decoded masks to keep in memory. Full-resolution
                    masks are large; raise this only with the RAM to match.
    """

    def __init__(self, encodings: dict[str, str], cache_size: int = 2) -> None:
        self._encodings = dict(encodings)
        self._cache_size = max(1, cache_size)
        self._cache: dict[str, NDArray[np.uint8]] = {}

    @classmethod
    def from_csv(
        cls,
        csv_path: Path | str,
        id_column: str = "id",
        encoding_column: str = "encoding",
        cache_size: int = 2,
    ) -> RLEMaskProvider:
        """Build a provider from a two-column ``id,encoding`` CSV.

        Args:
            csv_path:        CSV file listing one RLE string per slide.
            id_column:       Column holding the slide identifier.
            encoding_column: Column holding the RLE string.
            cache_size:      Decoded masks to keep in memory.

        Returns:
            A provider covering every row of the CSV.
        """
        import pandas as pd  # noqa: PLC0415

        df = pd.read_csv(csv_path)
        missing = {id_column, encoding_column} - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing column(s): {sorted(missing)}")

        encodings = dict(zip(df[id_column].astype(str), df[encoding_column], strict=True))
        logger.info("Loaded %d RLE encodings from %s", len(encodings), csv_path)
        return cls(encodings, cache_size=cache_size)

    def slide_ids(self) -> list[str]:
        return list(self._encodings)

    def get_mask(self, slide_id: str, shape: tuple[int, int]) -> NDArray[np.uint8]:
        cached = self._cache.get(slide_id)
        if cached is not None:
            return cached

        mask = rle_decode(self._encodings[slide_id], shape)

        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[slide_id] = mask
        return mask


# ── Dataset ───────────────────────────────────────────────────────────────────


class SegmentationDataset(Dataset):
    """Map-style dataset yielding ``{"image", "mask"}`` patches read from WSIs.

    Args:
        slide_dir:      Directory holding the slide files.
        mask_provider:  Source of ground-truth masks.
        tiling_cfg:     Overlap and tissue-filter thresholds.
        patch_size:     Patch side length (H = W) at *target_mpp*.
        target_mpp:     Resolution to train at, in µm/px. Slides without MPP
                        metadata (HuBMAP among them) are read at level 0.
        slide_ids:      Restrict to these slides — this is how a train/val split
                        is applied. Defaults to everything the provider covers.
        transform:      albumentations transform applied to image and mask.
        extensions:     Slide file extensions to try, in priority order.
        thumbnail_size: Overview size used for the tissue pre-filter.
    """

    def __init__(
        self,
        slide_dir: Path | str,
        mask_provider: MaskProvider,
        tiling_cfg: TilingConfig,
        patch_size: int = 512,
        target_mpp: float = 0.5,
        slide_ids: Sequence[str] | None = None,
        transform: Callable[..., dict[str, np.ndarray]] | None = None,
        extensions: Sequence[str] = _DEFAULT_SLIDE_EXTENSIONS,
        thumbnail_size: tuple[int, int] = (2048, 2048),
    ) -> None:
        self.slide_dir = Path(slide_dir)
        self.mask_provider = mask_provider
        self.tiling_cfg = tiling_cfg
        self.patch_size = patch_size
        self.target_mpp = target_mpp
        self.transform = transform
        self._extensions = tuple(extensions)
        self._thumbnail_size = thumbnail_size

        available = set(mask_provider.slide_ids())
        self.slide_ids: list[str] = (
            [s for s in slide_ids if s in available] if slide_ids is not None else sorted(available)
        )

        self._reader: BaseWSIReader | None = None
        self._reader_slide: str | None = None
        self._dimensions: dict[str, tuple[int, int]] = {}
        self.patch_list: list[tuple[str, PatchCoord]] = []
        self._build_patch_index()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _build_patch_index(self) -> None:
        """Tile every slide and keep the coords whose thumbnail region has tissue."""
        pbar = create_progress_bar(len(self.slide_ids), desc="Indexing slides", unit="slide")

        for slide_id in self.slide_ids:
            pbar.update(1)
            slide_path = self.slide_path(slide_id)
            if not slide_path.exists():
                logger.warning("Slide not found, skipping: %s", slide_path)
                continue

            try:
                with get_reader(slide_path) as reader:
                    metadata = reader.get_metadata()
                    thumbnail = reader.get_thumbnail(self._thumbnail_size)

                self._dimensions[slide_id] = metadata.dimensions

                coords = generate_patch_grid(
                    metadata,
                    patch_size=self.patch_size,
                    target_mpp=self.target_mpp,
                    tiling_cfg=self.tiling_cfg,
                    slide_id=slide_id,
                )
                tissue_coords = filter_coords_by_tissue(
                    coords, thumbnail, metadata.dimensions, self.tiling_cfg
                )
                self.patch_list.extend((slide_id, c) for c in tissue_coords)
                logger.info(
                    "%s: %d/%d patches passed the tissue filter",
                    slide_id,
                    len(tissue_coords),
                    len(coords),
                )
            except Exception:
                logger.exception("Failed to index slide %s", slide_id)

        pbar.close()
        logger.info(
            "Patch index built: %d patches from %d slides",
            len(self.patch_list),
            len(self.slide_ids),
        )

    def subset(self, n: int) -> None:
        """Truncate the patch index to *n* entries (debug / smoke-test runs)."""
        self.patch_list = self.patch_list[:n]

    # ── Slide access ──────────────────────────────────────────────────────────

    def slide_path(self, slide_id: str) -> Path:
        """Resolve *slide_id* to a file, trying each configured extension."""
        for ext in self._extensions:
            candidate = self.slide_dir / f"{slide_id}{ext}"
            if candidate.exists():
                return candidate
        return self.slide_dir / f"{slide_id}{self._extensions[0]}"

    def _get_reader(self, slide_id: str) -> BaseWSIReader:
        """Return an open reader for *slide_id*, reusing the last one opened.

        Patches are indexed slide by slide, so within a worker consecutive items
        usually hit the same slide and the handle is reused.
        """
        if self._reader is not None and self._reader_slide == slide_id:
            return self._reader

        self._close_reader()
        reader = get_reader(self.slide_path(slide_id))
        reader.open()
        self._reader = reader
        self._reader_slide = slide_id
        return reader

    def _close_reader(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass
        self._reader = None
        self._reader_slide = None

    def __del__(self) -> None:
        self._close_reader()

    def _slide_dimensions(self, slide_id: str) -> tuple[int, int]:
        if slide_id not in self._dimensions:
            self._dimensions[slide_id] = self._get_reader(slide_id).get_metadata().dimensions
        return self._dimensions[slide_id]

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.patch_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        slide_id, coord = self.patch_list[idx]
        reader = self._get_reader(slide_id)

        image = reader.read_region(
            location=(coord.x, coord.y),
            level=coord.level,
            size=(coord.patch_size, coord.patch_size),
        )
        image = ensure_rgb(image)
        image = rescale_patch(image, coord, self.patch_size)
        image = pad_to_size(image, self.patch_size)

        mask = self._read_mask_region(slide_id, coord)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255.0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return {"image": image_t, "mask": mask_t}

    def _read_mask_region(self, slide_id: str, coord: PatchCoord) -> NDArray[np.uint8]:
        """Crop the slide mask to *coord* and align it with the model patch size.

        The provider's mask is at level-0 resolution, so the crop uses the
        coord's level-0 extent and is then resized to ``patch_size`` — the same
        rescale the image goes through.
        """
        width, height = self._slide_dimensions(slide_id)
        full_mask = self.mask_provider.get_mask(slide_id, (height, width))

        extent = max(1, round(coord.patch_size * coord.rescale_factor))
        crop = full_mask[coord.y : coord.y + extent, coord.x : coord.x + extent]

        if crop.shape[:2] != (extent, extent):
            crop = np.pad(
                crop,
                ((0, extent - crop.shape[0]), (0, extent - crop.shape[1])),
                mode="constant",
            )

        if extent != self.patch_size:
            import cv2  # noqa: PLC0415

            crop = cv2.resize(
                crop,
                (self.patch_size, self.patch_size),
                interpolation=cv2.INTER_NEAREST,
            )

        return crop.astype(np.uint8)


# ── Pre-extracted tiles ───────────────────────────────────────────────────────


class PatchDirectoryDataset(Dataset):
    """Patch-level segmentation dataset for H&E histology.

    Args:
        image_dir:   Directory containing patch images.
        mask_dir:    Directory containing binary mask images (same stems).
        transform:   albumentations transform applied to both image and mask.
        extensions:  Accepted file extensions.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transform: Callable[..., dict[str, np.ndarray]] | None = None,
        extensions: set | None = None,
    ) -> None:
        self._image_dir = Path(image_dir)
        self._mask_dir = Path(mask_dir)
        self._transform = transform
        self._ext = extensions or _TILE_EXTENSIONS

        self._samples: list[tuple[Path, Path]] = self._find_samples()
        logger.info("PatchDirectoryDataset: %d patches in %s", len(self._samples), image_dir)

    def _find_samples(self) -> list[tuple[Path, Path]]:
        pairs = []
        for img_path in sorted(self._image_dir.iterdir()):
            if img_path.suffix.lower() not in self._ext:
                continue
            mask_path = self._mask_dir / img_path.name
            if not mask_path.exists():
                # Try other extensions
                for ext in self._ext:
                    alt = self._mask_dir / (img_path.stem + ext)
                    if alt.exists():
                        mask_path = alt
                        break
                else:
                    logger.warning("No mask found for %s — skipping.", img_path.name)
                    continue
            pairs.append((img_path, mask_path))
        return pairs

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        from PIL import Image  # noqa: PLC0415

        img_path, mask_path = self._samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 127).astype(np.uint8)

        if self._transform is not None:
            augmented = self._transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": image_t, "mask": mask_t, "path": str(img_path)}


def build_train_dataloader(
    image_dir: Path,
    mask_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    transform: Callable[..., dict[str, np.ndarray]] | None = None,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for segmentation training."""
    dataset = PatchDirectoryDataset(image_dir, mask_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


HistoSegDataset = PatchDirectoryDataset
"""Deprecated alias for :class:`PatchDirectoryDataset`.

Renamed because "HistoSeg" said nothing about what distinguishes it from
:class:`SegmentationDataset`; the new name names the difference — it reads
patches from a directory. Kept so existing imports keep working.
"""
