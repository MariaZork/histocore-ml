"""Training dataset for histology segmentation.

Loads (patch, mask) pairs from a directory of pre-extracted tiles.
Supports optional albumentations augmentation pipeline.

Expected directory layout::

    data/
      images/
        slide_001_x0_y0.png
        slide_001_x0_y512.png
        ...
      masks/
        slide_001_x0_y0.png
        slide_001_x0_y512.png
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS = {".png", ".jpg", ".tif", ".tiff"}


class HistoSegDataset(Dataset):
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
        transform=None,
        extensions: Optional[set] = None,
    ) -> None:
        self._image_dir = Path(image_dir)
        self._mask_dir  = Path(mask_dir)
        self._transform = transform
        self._ext = extensions or _DEFAULT_EXTENSIONS

        self._samples: List[Tuple[Path, Path]] = self._find_samples()
        logger.info("HistoSegDataset: %d patches in %s", len(self._samples), image_dir)

    def _find_samples(self) -> List[Tuple[Path, Path]]:
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

    def __getitem__(self, idx: int):
        from PIL import Image  # noqa: PLC0415
        img_path, mask_path = self._samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(mask_path).convert("L"),  dtype=np.uint8)
        mask  = (mask > 127).astype(np.uint8)

        if self._transform is not None:
            augmented = self._transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        mask_t  = torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": image_t, "mask": mask_t, "path": str(img_path)}


def build_train_dataloader(
    image_dir: Path,
    mask_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    transform=None,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for segmentation training."""
    dataset = HistoSegDataset(image_dir, mask_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
