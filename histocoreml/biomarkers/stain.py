"""H&E / DAB stain separation and Ki-67 proliferation index."""

from __future__ import annotations

import numpy as np

# Standard H&E stain matrix (Ruifrok & Johnston, 2001)
_HE_MATRIX = np.array([
    [0.6500286, 0.7049342, 0.2860990],
    [0.2746924, 0.8736234, 0.4000313],
    [0.7138408, 0.0480177, 0.6991889],
])

# H-DAB stain matrix
_HDAB_MATRIX = np.array([
    [0.6500286, 0.7049342, 0.2860990],
    [0.2697800, 0.5731133, 0.7748341],
    [0.7138408, 0.0480177, 0.6991889],
])


def separate_he_channels(
    patch: np.ndarray,
    Io: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate H&E stained patch into Haematoxylin and Eosin channels.

    Args:
        patch: uint8 RGB array (H, W, 3).
        Io:    Transmitted light intensity (default 240).

    Returns:
        Tuple of (hematoxylin, eosin) float32 arrays (H, W) in [0, 1].
    """
    h, w = patch.shape[:2]
    img = patch.reshape(-1, 3).astype(np.float64)
    img = img.clip(1)
    od = -np.log(img / Io)

    stain_inv = np.linalg.pinv(_HE_MATRIX[:2])
    concentrations = od @ stain_inv
    concentrations = concentrations.clip(0)

    # Normalise each channel
    def _norm(c: np.ndarray) -> np.ndarray:
        c_max = c.max() or 1.0
        return (c / c_max).reshape(h, w).astype(np.float32)

    return _norm(concentrations[:, 0]), _norm(concentrations[:, 1])


def separate_hdab_channels(
    patch: np.ndarray,
    Io: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate H-DAB stained patch into Haematoxylin and DAB channels.

    DAB (3,3'-diaminobenzidine) is used in IHC staining (e.g. Ki-67, ER, PR).

    Args:
        patch: uint8 RGB array (H, W, 3).
        Io:    Transmitted light intensity.

    Returns:
        Tuple of (hematoxylin, dab) float32 arrays (H, W) in [0, 1].
    """
    h, w = patch.shape[:2]
    img = patch.reshape(-1, 3).astype(np.float64).clip(1)
    od = -np.log(img / Io)

    stain_inv = np.linalg.pinv(_HDAB_MATRIX[:2])
    concentrations = (od @ stain_inv).clip(0)

    def _norm(c: np.ndarray) -> np.ndarray:
        c_max = c.max() or 1.0
        return (c / c_max).reshape(h, w).astype(np.float32)

    return _norm(concentrations[:, 0]), _norm(concentrations[:, 1])


def compute_ki67_index(
    patch: np.ndarray,
    nuclei_mask: np.ndarray,
    dab_threshold: float = 0.2,
) -> float:
    """Estimate the Ki-67 proliferation index from an IHC-stained patch.

    Counts the fraction of nuclei with DAB (brown) staining above threshold.

    Args:
        patch:          uint8 RGB array (H, W, 3) — H-DAB stained IHC section.
        nuclei_mask:    Binary uint8 mask (H, W) marking nucleus pixels.
        dab_threshold:  DAB channel intensity threshold for positive staining.

    Returns:
        Ki-67 index as a float in [0, 1] (fraction of positive nuclei).
    """
    _, dab = separate_hdab_channels(patch)

    if nuclei_mask.sum() == 0:
        return float("nan")

    dab_in_nuclei = dab[nuclei_mask > 0]
    positive = (dab_in_nuclei >= dab_threshold).mean()
    return float(positive)
