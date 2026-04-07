"""Pure-numpy patch preprocessing helpers.

MPP enforcement
---------------
:func:`rescale_patch` corrects for any residual MPP error between the pyramid
level and the model's target resolution.

Stain normalisation
-------------------
:func:`macenko_normalise` applies Macenko H&E normalisation — useful for
reducing scanner and staining batch effects before training or inference.

Tissue detection
----------------
:func:`is_tissue` excludes white glass background and black scanner borders.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from histocoreml.config import TilingConfig
from histocoreml.preprocessing.patch_coord import PatchCoord

logger = logging.getLogger(__name__)


# ── MPP rescaling ─────────────────────────────────────────────────────────────


def rescale_patch(
    patch: np.ndarray,
    coord: PatchCoord,
    model_patch_size: int | None = None,
) -> np.ndarray:
    """Resize *patch* to the model's canonical input size at the target MPP.

    Args:
        patch: Raw uint8 patch from the reader, shape ``(H, W, C)``.
        coord: Patch descriptor carrying ``rescale_factor``.
        model_patch_size: Canonical model input size.

    Returns:
        Uint8 array of shape ``(model_patch_size, model_patch_size, C)``.
    """
    rf = coord.rescale_factor
    if abs(rf - 1.0) < 1e-3:
        return patch

    if model_patch_size is not None:
        target = model_patch_size
    else:
        target = max(1, round(coord.patch_size / rf))

    interp = cv2.INTER_AREA if rf > 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(patch, (target, target), interpolation=interp)
    logger.debug(
        "rescale_patch: rf=%.4f %dx%d → %dx%d",
        rf,
        patch.shape[1],
        patch.shape[0],
        target,
        target,
    )
    return resized


# ── Padding ───────────────────────────────────────────────────────────────────


def pad_to_size(patch: np.ndarray, target_size: int) -> np.ndarray:
    """Zero-pad an edge patch smaller than *target_size*.

    Args:
        patch: uint8 array ``(H, W, C)`` where H or W may be < target_size.
        target_size: Desired side length in pixels.

    Returns:
        uint8 array ``(target_size, target_size, C)``.
    """
    h, w = patch.shape[:2]
    if h == target_size and w == target_size:
        return patch
    padded = np.zeros((target_size, target_size, patch.shape[2]), dtype=np.uint8)
    padded[:h, :w] = patch
    return padded


# ── Tissue detection ──────────────────────────────────────────────────────────


def is_tissue(patch: np.ndarray, cfg: TilingConfig) -> bool:
    """Return True if the patch contains sufficient foreground tissue.

    Excludes both white background (glass) and black background (scanner
    border outside the scanned region).

    Args:
        patch: uint8 RGB array ``(H, W, 3)``.
        cfg: Tiling configuration with tissue/background thresholds.

    Returns:
        ``True`` if the foreground pixel fraction ≥ ``cfg.tissue_threshold``.
    """
    gray = patch.mean(axis=-1)
    foreground_pixels = int(
        ((gray < cfg.background_value) & (gray > cfg.black_value)).sum()
    )
    total_pixels = gray.size
    required_pixels = int(cfg.tissue_threshold * total_pixels)
    return foreground_pixels >= required_pixels


def tissue_mask(patch: np.ndarray, cfg: TilingConfig) -> np.ndarray:
    """Return a boolean mask indicating tissue pixels.

    Args:
        patch: uint8 RGB array ``(H, W, 3)``.
        cfg: Tiling configuration.

    Returns:
        Boolean array ``(H, W)``.
    """
    gray = patch.mean(axis=-1)
    return (gray < cfg.background_value) & (gray > cfg.black_value)


# ── Stain normalisation ───────────────────────────────────────────────────────

# Macenko reference stain matrix (trained on TCGA H&E slides)
_MACENKO_REFERENCE = np.array(
    [
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ]
)

_MACENKO_MAX_C = np.array([1.9705, 1.0308])


def macenko_normalise(
    patch: np.ndarray,
    reference_stain_matrix: np.ndarray | None = None,
    reference_max_c: np.ndarray | None = None,
    Io: int = 240,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply Macenko H&E stain normalisation to a patch.

    Reduces batch effects from different scanners or staining protocols.

    Args:
        patch: uint8 RGB array ``(H, W, 3)``.
        reference_stain_matrix: 3×2 reference matrix. Uses built-in default
            (TCGA H&E) if None.
        reference_max_c: Reference concentration scale. Uses default
            if None.
        Io: Transmitted light intensity (default 240).
        beta: Threshold for excluding pixels with low OD.
        alpha: Percentile for concentration scaling.

    Returns:
        Normalised uint8 RGB array ``(H, W, 3)``.
    """
    HE_ref = (
        reference_stain_matrix
        if reference_stain_matrix is not None
        else _MACENKO_REFERENCE
    )
    maxC_ref = reference_max_c if reference_max_c is not None else _MACENKO_MAX_C

    h, w = patch.shape[:2]
    img: NDArray[np.float64] = patch.reshape(-1, 3).astype(np.float64)

    # Optical density
    img_od = -np.log(img.clip(1) / Io)

    # Remove background
    mask = (img_od >= beta).all(axis=1)
    if mask.sum() == 0:
        return patch  # no tissue — return unchanged

    od_hat = img_od[mask]

    # SVD
    _, _, V = np.linalg.svd(od_hat, full_matrices=False)
    V = V[:2].T  # first two principal components

    # Project onto plane
    that = od_hat @ V
    phi = np.arctan2(that[:, 1], that[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    v1 = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
    v2 = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

    # Order H and E
    if v1[0] > v2[0]:
        HE = np.array([v1, v2]).T
    else:
        HE = np.array([v2, v1]).T

    # Concentration
    Y: NDArray[np.float64] = np.reshape(img_od, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    maxC = np.array([np.percentile(C[0], 99), np.percentile(C[1], 99)])
    C2 = C / maxC[:, np.newaxis] * maxC_ref[:, np.newaxis]

    # Reconstruct
    Inorm = Io * np.exp(-HE_ref @ C2)
    Inorm = np.clip(Inorm.T, 0, 255).astype(np.uint8)
    return Inorm.reshape(h, w, 3)
