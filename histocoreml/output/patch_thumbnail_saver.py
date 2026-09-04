"""On-the-fly patch thumbnail saving for segmentation QC.

Each saved PNG contains three equal panels side-by-side on a dark canvas::

    [ raw patch image ] | [ binary mask (white=fg) ] | [ red overlay ]

A ``cov XX.X%`` label (black text) is drawn in the bottom-left corner.
A montage grid is assembled from small overlay cells and written once
after all patches have been processed.

Public API
----------
write_patch_thumbnail   Write one 3-panel PNG; return a small montage cell.
finalise_montage        Stitch collected cells into a grid PNG.
patch_to_rgb_uint8      Convert ``(C, H, W)`` float32 tensor patch → ``(H, W, 3)`` uint8.

Output layout::

    <output_dir>/
    └── patch_thumbnails/
        ├── <stem>_r0003_c0012.png
        └── ...
    <output_dir>/
    └── <stem>_montage.png

Usage::

    from histocoreml.output.patch_thumbnail_saver import (
        write_patch_thumbnail,
        finalise_montage,
        patch_to_rgb_uint8,
    )

    cells = []
    for image, mask, coord in patches:
        cell = write_patch_thumbnail(image, mask, coord, thumb_dir, stem="slide_001")
        if cell is not None:
            cells.append(cell)

    finalise_montage(cells, output_dir, stem="slide_001")
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from histocoreml.preprocessing.patch_coord import PatchCoord

logger = logging.getLogger(__name__)

# ── Tuneable defaults ─────────────────────────────────────────────────────────
_PANEL_SIZE: int = 256  # width & height of each panel (px)
_GAP: int = 6  # dark separator between panels (px)
_CANVAS_BG: int = 30  # dark background fill value
_MONTAGE_COLS: int = 5  # columns in the overview grid
_MAX_THUMBNAILS: int = 256  # hard cap per slide
_OVERLAY_ALPHA: float = 0.50  # opacity of the red foreground overlay
_LABEL_COLOR: tuple[int, int, int] = (0, 0, 0)  # black text for coverage label


# ── Public API ────────────────────────────────────────────────────────────────


def write_patch_thumbnail(
    image: np.ndarray,
    mask: np.ndarray,
    coord: PatchCoord,
    thumb_dir: Path,
    stem: str,
    panel_size: int = _PANEL_SIZE,
    max_thumbnails: int = _MAX_THUMBNAILS,
) -> np.ndarray | None:
    """Write a 3-panel QC PNG for one patch and return a small montage cell.

    The three panels are:

    1. **Raw image** — the original tissue patch (RGB).
    2. **Binary mask** — white = foreground, black = background.
    3. **Overlay** — raw image with a semi-transparent red foreground region.

    A ``cov XX.X%`` label (black text) is drawn on the bottom-left of the PNG.

    Args:
        image:          uint8 RGB ``(H, W, 3)`` — call :func:`patch_to_rgb_uint8`
                        before passing float tensor patches here.
        mask:           uint8 binary ``(H, W)`` with values 0 or 1.
        coord:          :class:`~histocoreml.preprocessing.patch_coord.PatchCoord`
                        grid position; used for the output filename.
        thumb_dir:      Directory to write the PNG into (must already exist).
        stem:           WSI filename stem used as a filename prefix.
        panel_size:     Side length (px) of each individual panel in the card.
        max_thumbnails: Skip writing if the directory already contains this many
                        PNGs (prevents flooding disk on large slides).

    Returns:
        Small square overlay cell ``(panel_size // 2, panel_size // 2, 3)``
        uint8 array for later montage assembly, or *None* on failure / cap reached.
    """
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        logger.warning("Pillow not available — skipping patch thumbnail.")
        return None

    existing = sum(1 for _ in thumb_dir.glob("*.png"))
    if existing >= max_thumbnails:
        return None

    try:
        p, g = panel_size, _GAP

        panel_img = _resize(image, p, p)
        panel_mask = _make_mask_panel(mask, p)
        panel_over = _make_overlay_panel(panel_img, mask, p)

        canvas_w = p * 3 + g * 2
        canvas: np.ndarray = np.full((p, canvas_w, 3), _CANVAS_BG, dtype=np.uint8)
        canvas[:, 0:p] = panel_img
        canvas[:, p + g : p * 2 + g] = panel_mask
        canvas[:, p * 2 + g * 2 : p * 3 + g * 2] = panel_over

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        coverage = float(mask.mean()) * 100
        draw.text((4, p - 14), f"cov {coverage:.1f}%", fill=_LABEL_COLOR)

        name = f"{stem}_r{coord.row_idx:04d}_c{coord.col_idx:04d}.png"
        pil.save(str(thumb_dir / name))

        # Return a small overlay cell for the montage
        cell_size = p // 2
        return _resize(panel_over, cell_size, cell_size)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Thumbnail failed r%d c%d: %s", coord.row_idx, coord.col_idx, exc)
        return None


def finalise_montage(
    cells: list[np.ndarray],
    output_dir: Path,
    stem: str,
    cols: int = _MONTAGE_COLS,
) -> Path | None:
    """Stitch small overlay cells into a single overview grid PNG.

    Args:
        cells:      List of ``(cell_h, cell_w, 3)`` uint8 arrays returned by
                    :func:`write_patch_thumbnail`. All cells must be the same size.
        output_dir: Directory where the montage file is written.
        stem:       Filename stem prefix (``<stem>_montage.png``).
        cols:       Number of columns in the grid (default 5).

    Returns:
        :class:`Path` to the written montage, or *None* on failure / empty input.
    """
    if not cells:
        return None

    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        logger.warning("Pillow not available — skipping montage.")
        return None

    try:
        ch, cw = cells[0].shape[:2]
        n = len(cells)
        rows = math.ceil(n / cols)

        montage = np.full((rows * ch, cols * cw, 3), _CANVAS_BG, dtype=np.uint8)
        for i, cell in enumerate(cells):
            r, c = divmod(i, cols)
            montage[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = cell

        out = output_dir / f"{stem}_montage.png"
        Image.fromarray(montage).save(str(out))
        logger.info("Saved montage (%d patches, %d×%d grid) → %s", n, rows, cols, out)
        return out

    except Exception as exc:  # noqa: BLE001
        logger.error("Montage failed: %s", exc)
        return None


def patch_to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a patch array to a displayable uint8 RGB image.

    Handles the following input formats:

    * ``(C, H, W)`` float32 in ``[0, 1]`` — PyTorch channel-first (most common)
    * ``(H, W, C)`` uint8 or float        — channel-last (already converted)
    * ``(H, W)``    any dtype             — grayscale

    Args:
        img: Array of shape ``(C, H, W)``, ``(H, W, C)``, or ``(H, W)``.

    Returns:
        uint8 array of shape ``(H, W, 3)``.
    """
    # (C, H, W) → (H, W, C):  C is always ≤ 4; spatial dims are large
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] > 4:
        img = np.moveaxis(img, 0, -1)

    # float [0, 1] → uint8 [0, 255]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    # Ensure 3-channel RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]  # drop alpha

    return img


# ── Private helpers ───────────────────────────────────────────────────────────


def _resize(arr: np.ndarray, h: int, w: int, nearest: bool = False) -> np.ndarray:
    """Resize *arr* to ``(h, w[, C])`` using Pillow."""
    from PIL import Image  # noqa: PLC0415

    mode = "L" if arr.ndim == 2 else "RGB"
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.array(Image.fromarray(arr, mode=mode).resize((w, h), resample))


def _make_mask_panel(mask: np.ndarray, size: int) -> np.ndarray:
    """Return a ``(size, size, 3)`` uint8 panel: white=fg, black=bg."""
    gray: np.ndarray = (mask * 255).astype(np.uint8)
    small = _resize(gray, size, size, nearest=True)
    return np.stack([small] * 3, axis=-1)


def _make_overlay_panel(panel_img: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    """Blend a semi-transparent red region over *panel_img* where mask == 1."""
    overlay = panel_img.copy()
    fg: np.ndarray = _resize(mask.astype(np.uint8), size, size, nearest=True).astype(bool)
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[fg] = (
        (1 - _OVERLAY_ALPHA) * overlay[fg].astype(np.float32)
        + _OVERLAY_ALPHA * red[fg].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def visualise_dataset_samples(
    dataset: object,
    output_dir: Path,
    stem: str = "samples",
    n_samples: int = 6,
    max_scanned: int = 200,
    cols: int = 3,
) -> Path | None:
    """Write QC thumbnails for a handful of dataset samples, plus a montage.

    Intended for training datasets yielding ``{"image", "mask"}`` items: run it
    before epoch 1 to confirm patches and masks actually line up, and again
    periodically to watch what the model is being fed.

    Samples that are almost entirely empty are skipped, so a slide with sparse
    tissue still produces a useful sheet.

    Args:
        dataset:     Any indexable yielding ``{"image": Tensor|ndarray,
                     "mask": Tensor|ndarray}``.
        output_dir:  Root directory; files land in ``<output_dir>/sample_visualizations``.
        stem:        Filename prefix, e.g. ``"epoch_005"``.
        n_samples:   Number of samples to write.
        max_scanned: Give up looking for non-empty samples after this many items.
        cols:        Montage grid columns.

    Returns:
        Path to the montage PNG, or *None* if nothing could be visualised.
    """
    viz_dir = Path(output_dir) / "sample_visualizations"
    # One directory per call: write_patch_thumbnail caps on the file count
    # already present, so a shared directory would silently stop producing
    # thumbnails after the first call.
    thumb_dir = viz_dir / "thumbnails" / stem
    thumb_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)  # type: ignore[arg-type]
    if total == 0:
        logger.warning("Dataset is empty — nothing to visualise.")
        return None

    cells: list[np.ndarray] = []
    for idx in range(min(total, max_scanned)):
        if len(cells) >= n_samples:
            break

        sample = dataset[idx]  # type: ignore[index]
        image = _to_numpy(sample["image"])
        mask = _to_numpy(sample["mask"])

        image_uint8 = patch_to_rgb_uint8(image)
        if image_uint8.max() == 0:
            continue  # all-black patch, nothing to look at

        mask_binary = (mask.squeeze() > 0.5).astype(np.uint8)

        coord = PatchCoord(
            x=0,
            y=0,
            level=0,
            patch_size=image_uint8.shape[0],
            col_idx=idx,
            row_idx=len(cells),
            slide_id=stem,
        )
        cell = write_patch_thumbnail(
            image=image_uint8,
            mask=mask_binary,
            coord=coord,
            thumb_dir=thumb_dir,
            stem=f"{stem}_sample",
            max_thumbnails=n_samples,
        )
        if cell is not None:
            cells.append(cell)

    if not cells:
        logger.warning("No non-empty samples found for visualisation.")
        return None

    montage_path = finalise_montage(cells, viz_dir, stem, cols=cols)
    logger.info("Saved %d sample thumbnails → %s", len(cells), thumb_dir)
    return montage_path


def _to_numpy(value: object) -> np.ndarray:
    """Detach a tensor to numpy, or pass an array through unchanged."""
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(value)
