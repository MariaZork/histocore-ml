"""Loss functions for histology segmentation training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary and multi-class segmentation."""

    def __init__(self, smooth: float = 1.0, from_logits: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()
        intersection = (pred * target).sum()
        return 1 - (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class DiceBCELoss(nn.Module):
    """Combination of Dice loss and Binary Cross-Entropy."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice = DiceLoss(from_logits=True)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(pred, target.float())
        dice = self.dice(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy examples to focus on hard ones."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        prob = torch.sigmoid(pred)
        p_t  = prob * target + (1 - prob) * (1 - target)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss — generalises Dice with separate FP / FN weights.

    Set ``alpha > beta`` to penalise false negatives more (e.g. for recall-
    oriented tumour detection).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred).view(-1)
        target = target.float().view(-1)
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


def get_loss(name: str) -> nn.Module:
    """Factory for loss functions by name."""
    registry = {
        "dice":     DiceLoss,
        "bce":      lambda: nn.BCEWithLogitsLoss(),
        "dice_bce": DiceBCELoss,
        "focal":    FocalLoss,
        "tversky":  TverskyLoss,
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(registry)}")
    return registry[name]()
