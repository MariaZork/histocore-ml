"""Evaluation metrics for segmentation."""

from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred   = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (Jaccard index)."""
    pred   = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return (intersection + smooth) / (union + smooth)


def precision_recall_f1(
    pred: np.ndarray, target: np.ndarray
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) for binary masks."""
    pred   = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    tp = (pred & target).sum()
    fp = (pred & ~target).sum()
    fn = (~pred & target).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th-percentile Hausdorff distance between two binary masks."""
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415
    pred   = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() or not target.any():
        return float("inf")
    dt_pred   = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)
    hd_pred_to_gt = dt_target[pred].max()
    hd_gt_to_pred = dt_pred[target].max()
    return float(max(hd_pred_to_gt, hd_gt_to_pred))
