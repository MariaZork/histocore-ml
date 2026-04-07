"""Evaluation metrics for segmentation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred_flat: NDArray[np.bool_] = pred.astype(bool).ravel()
    target_flat: NDArray[np.bool_] = target.astype(bool).ravel()
    intersection = (pred_flat & target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (Jaccard index)."""
    pred_flat: NDArray[np.bool_] = pred.astype(bool).ravel()
    target_flat: NDArray[np.bool_] = target.astype(bool).ravel()
    intersection = (pred_flat & target_flat).sum()
    union = (pred_flat | target_flat).sum()
    return (intersection + smooth) / (union + smooth)


def precision_recall_f1(
    pred: np.ndarray, target: np.ndarray
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) for binary masks."""
    pred_flat: NDArray[np.bool_] = pred.astype(bool).ravel()
    target_flat: NDArray[np.bool_] = target.astype(bool).ravel()
    tp = (pred_flat & target_flat).sum()
    fp = (pred_flat & ~target_flat).sum()
    fn = (~pred_flat & target_flat).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th-percentile Hausdorff distance between two binary masks."""
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415
    pred_bool: NDArray[np.bool_] = pred.astype(bool)
    target_bool: NDArray[np.bool_] = target.astype(bool)
    if not pred_bool.any() or not target_bool.any():
        return float("inf")
    dt_pred = distance_transform_edt(~pred_bool)
    dt_target = distance_transform_edt(~target_bool)
    hd_pred_to_gt = dt_target[pred_bool].max()
    hd_gt_to_pred = dt_pred[target_bool].max()
    return float(max(hd_pred_to_gt, hd_gt_to_pred))
