"""Nuclei detection and morphology measurement from H&E patches."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def detect_nuclei(
    patch: np.ndarray,
    min_area: int = 50,
    max_area: int = 5000,
) -> tuple[np.ndarray, list[dict]]:
    """Detect nuclei in an H&E patch using colour thresholding + watershed.

    Args:
        patch: uint8 RGB array (H, W, 3).
        min_area: Minimum nucleus area in pixels.
        max_area: Maximum nucleus area in pixels.

    Returns:
        Tuple of (labelled_mask, list_of_nucleus_dicts).
        Each nucleus dict has keys: label, centroid, area, bbox.
    """
    import cv2  # noqa: PLC0415

    # Check for uniform patches (all same color) - no nuclei possible
    if np.all(patch == patch[0, 0, 0]):
        return np.zeros(patch.shape[:2], dtype=np.int32), []

    # Convert to LAB and threshold the 'a' channel (red-green axis → nuclei are purple)
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1]

    _, binary = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform + watershed
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg: NDArray[np.uint8] = sure_fg.astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(sure_fg)
    labels: NDArray[np.int32] = labels + 1
    labels[binary == 0] = 0

    # Filter by area
    nuclei = []
    for lbl in range(1, labels.max() + 1):
        region = labels == lbl
        area = int(region.sum())
        if not (min_area <= area <= max_area):
            labels[region] = 0
            continue
        ys, xs = np.where(region)
        centroid = (float(xs.mean()), float(ys.mean()))
        nuclei.append(
            {
                "label": lbl,
                "centroid": centroid,
                "area": area,
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            }
        )

    return labels.astype(np.int32), nuclei


def measure_nuclei_morphology(
    patch: np.ndarray,
    labelled_mask: np.ndarray,
) -> list[dict]:
    """Compute morphological features for each labelled nucleus.

    Args:
        patch: uint8 RGB array (H, W, 3).
        labelled_mask: int32 label array (H, W), 0 = background.

    Returns:
        List of dicts with keys: label, area, eccentricity, solidity,
        mean_hematoxylin, mean_eosin, circularity.
    """
    try:
        from skimage.measure import regionprops  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scikit-image is required: pip install scikit-image") from exc

    from histocoreml.biomarkers.stain import (  # noqa: PLC0415
        separate_he_channels,
    )

    h_channel, _ = separate_he_channels(patch)

    features = []
    for props in regionprops(labelled_mask):
        lbl = props.label
        region = labelled_mask == lbl
        area = props.area
        perimeter = props.perimeter + 1e-8
        circularity = 4 * np.pi * area / (perimeter**2)

        features.append(
            {
                "label": lbl,
                "area": area,
                "eccentricity": float(props.eccentricity),
                "solidity": float(props.solidity),
                "circularity": float(circularity),
                "mean_hematoxylin": (float(h_channel[region].mean()) if region.any() else 0.0),
            }
        )

    return features
