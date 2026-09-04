"""Run-Length Encoding (RLE) codec for binary masks.

Two RLE flavours
----------------
``PlainRLE`` — Row-major (value, run_length) pairs. Compact JSON.
``CocoRLE`` — Column-major run-lengths compatible with ``pycocotools.mask``.

Streaming helpers
-----------------
``encode_patches_to_plain`` / ``merge_plain_rles``
  Compress patch predictions immediately after the forward pass and merge
  into a full-slide mask without materialising the full uncompressed canvas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlainRLE:
    """Row-major RLE of a binary mask."""

    shape: tuple[int, int]
    runs: list[tuple[int, int]]

    def decode(self) -> np.ndarray:
        return plain_rle_decode(self)

    def compressed_size(self) -> int:
        return len(self.runs) * 8

    def compression_ratio(self) -> float:
        original = self.shape[0] * self.shape[1]
        return original / max(self.compressed_size(), 1)

    def total_foreground(self) -> int:
        return sum(count for value, count in self.runs if value == 1)


@dataclass(frozen=True)
class CocoRLE:
    """COCO-style column-major RLE of a binary mask."""

    size: tuple[int, int]
    counts: list[int]

    def to_coco_dict(self) -> dict:
        return {"size": list(self.size), "counts": self.counts}

    def decode(self) -> np.ndarray:
        return coco_rle_decode(self)

    def total_foreground(self) -> int:
        return sum(c for i, c in enumerate(self.counts) if i % 2 == 1)


# ── Plain RLE ─────────────────────────────────────────────────────────────────


def plain_rle_encode(mask: np.ndarray) -> PlainRLE:
    """Encode a binary mask as row-major plain RLE."""
    _validate_mask(mask)
    flat: NDArray[np.uint8] = mask.ravel()
    if flat.size == 0:
        return PlainRLE(shape=mask.shape, runs=[])
    change_pos = np.where(np.diff(flat))[0] + 1
    starts = np.concatenate(([0], change_pos))
    ends = np.concatenate((change_pos, [flat.size]))
    runs: list[tuple[int, int]] = [
        (int(flat[s]), int(e - s)) for s, e in zip(starts, ends, strict=True)
    ]
    return PlainRLE(shape=mask.shape, runs=runs)


def plain_rle_decode(rle: PlainRLE) -> np.ndarray:
    """Reconstruct a binary mask from :class:`PlainRLE`."""
    h, w = rle.shape
    flat: NDArray[np.uint8] = np.empty(h * w, dtype=np.uint8)
    pos = 0
    for value, count in rle.runs:
        flat[pos : pos + count] = value
        pos += count
    return flat.reshape(h, w)


# ── COCO RLE ──────────────────────────────────────────────────────────────────


def coco_rle_encode(mask: np.ndarray) -> CocoRLE:
    """Encode a binary mask as COCO-style column-major RLE."""
    _validate_mask(mask)
    flat: NDArray[np.uint8] = mask.ravel(order="F")
    if flat.size == 0:
        return CocoRLE(size=mask.shape, counts=[])
    change_pos = np.where(np.diff(flat))[0] + 1
    starts = np.concatenate(([0], change_pos))
    ends = np.concatenate((change_pos, [flat.size]))
    lengths = (ends - starts).tolist()
    if int(flat[0]) == 1:
        lengths = [0] + lengths
    return CocoRLE(size=mask.shape, counts=lengths)


def coco_rle_decode(rle: CocoRLE) -> np.ndarray:
    """Reconstruct a binary mask from :class:`CocoRLE`."""
    h, w = rle.size
    flat: NDArray[np.uint8] = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(rle.counts):
        if i % 2 == 1:
            flat[pos : pos + count] = 1
        pos += count
    return flat.reshape(h, w, order="F")


# ── Streaming encode / merge ──────────────────────────────────────────────────


def encode_patches_to_plain(masks: np.ndarray) -> list[PlainRLE]:
    """Encode a batch of patch masks to :class:`PlainRLE` objects."""
    return [plain_rle_encode(masks[i]) for i in range(len(masks))]


def merge_plain_rles(
    patch_rles: Sequence[tuple],
    canvas_height: int,
    canvas_width: int,
    inf_ds: float = 1.0,
    downsample_factor: int = 1,
) -> np.ndarray:
    """Reconstruct a full-slide binary mask from patch-level PlainRLE objects."""
    canvas: NDArray[np.uint8] = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    scale = inf_ds * downsample_factor
    for coord, rle in patch_rles:
        patch_mask = rle.decode()
        ph, pw = patch_mask.shape
        cx = int(coord.x / scale)
        cy = int(coord.y / scale)
        y0, x0 = max(cy, 0), max(cx, 0)
        y1 = min(cy + ph, canvas_height)
        x1 = min(cx + pw, canvas_width)
        if y1 <= y0 or x1 <= x0:
            continue
        my0, mx0 = y0 - cy, x0 - cx
        canvas[y0:y1, x0:x1] |= patch_mask[my0 : my0 + (y1 - y0), mx0 : mx0 + (x1 - x0)]
    return canvas


# ── Serialisation ─────────────────────────────────────────────────────────────


def plain_rle_to_dict(rle: PlainRLE) -> dict:
    return {"shape": list(rle.shape), "runs": [list(r) for r in rle.runs]}


def plain_rle_from_dict(d: dict) -> PlainRLE:
    h, w = d["shape"]
    runs = [tuple(r) for r in d["runs"]]
    return PlainRLE(shape=(h, w), runs=runs)


def coco_rle_from_dict(d: dict) -> CocoRLE:
    seg = d.get("segmentation", d)
    h, w = seg["size"]
    return CocoRLE(size=(h, w), counts=list(seg["counts"]))


# ── Simple API (compatible with HuBMAP/other competitions) ────────────────────


def rle_encode(mask: np.ndarray) -> str:
    """Encode binary mask to RLE string (row-major, HuBMAP/Kaggle format).

    Args:
        mask: Binary mask (H, W) with values 0 or 1

    Returns:
        RLE string: "start length start length ..."

    Example::

        from histocoreml.output.rle_codec import rle_encode, rle_decode

        rle = rle_encode(mask)  # "1 20 30 5 ..."
        mask = rle_decode(rle, (512, 512))
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode RLE-encoded mask string to binary mask.

    Args:
        mask_rle: RLE string in "start length start length ..." format
        shape: (height, width) of output mask

    Returns:
        Binary mask (H, W) with values 0 or 1
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    img: NDArray[np.uint8] = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends, strict=True):
        img[lo:hi] = 1

    # HuBMAP RLE encodes pixels in column-major (Fortran) order:
    # pixels run top→bottom first, then left→right.
    # reshape with order="F" reconstructs (height, width) correctly.
    return img.reshape(shape, order="F")


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_mask(mask: np.ndarray) -> None:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.size > 0 and int(mask.max()) > 1:
        raise ValueError(
            f"mask values must be 0 or 1, got max={int(mask.max())}. "
            "Convert first: (mask > threshold).astype(np.uint8)"
        )
