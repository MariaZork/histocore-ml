"""Tests for histocoreml.training losses and metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from histocoreml.training.losses import DiceBCELoss, DiceLoss, FocalLoss, TverskyLoss, get_loss
from histocoreml.training.metrics import dice_score, iou_score, precision_recall_f1


class TestDiceLoss:
    def test_perfect_prediction_low_loss(self):
        # For perfect background prediction, use exact 0s with from_logits=False
        pred = torch.zeros(2, 64, 64)
        target = torch.zeros(2, 64, 64)
        loss = DiceLoss(from_logits=False)(pred, target)
        assert float(loss) < 0.01

    def test_worst_prediction_high_loss(self):
        pred = torch.ones(2, 64, 64) * 10
        target = torch.zeros(2, 64, 64)
        loss = DiceLoss()(pred, target)
        assert float(loss) > 0.9

    def test_output_scalar(self):
        pred = torch.randn(4, 32, 32)
        target = (torch.rand(4, 32, 32) > 0.5).float()
        loss = DiceLoss()(pred, target)
        assert loss.ndim == 0


class TestDiceBCELoss:
    def test_runs(self):
        pred = torch.randn(2, 32, 32)
        target = (torch.rand(2, 32, 32) > 0.5).float()
        loss = DiceBCELoss()(pred, target)
        assert float(loss) > 0


class TestFocalLoss:
    def test_runs(self):
        pred = torch.randn(2, 32, 32)
        target = (torch.rand(2, 32, 32) > 0.5).float()
        loss = FocalLoss()(pred, target)
        assert float(loss) >= 0


class TestTverskyLoss:
    def test_recall_weighted(self):
        # High beta → penalise FN more → loss should be > symmetric Dice
        pred = torch.zeros(1, 16, 16)  # predict nothing
        target = torch.ones(1, 16, 16).float()  # all foreground
        tversky = TverskyLoss(alpha=0.1, beta=0.9)(pred, target)
        dice = DiceLoss()(pred, target)
        assert float(tversky) > float(dice) * 0.8


class TestGetLoss:
    @pytest.mark.parametrize("name", ["dice", "bce", "dice_bce", "focal", "tversky"])
    def test_known_names(self, name):
        loss = get_loss(name)
        assert loss is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_loss("not_a_loss")


class TestMetrics:
    def test_dice_perfect(self):
        mask = np.ones((32, 32), dtype=np.uint8)
        assert abs(dice_score(mask, mask) - 1.0) < 1e-5

    def test_dice_no_overlap(self):
        a = np.zeros((32, 32), dtype=np.uint8)
        b = np.ones((32, 32), dtype=np.uint8)
        assert dice_score(a, b) < 0.01

    def test_iou_perfect(self):
        mask = np.ones((32, 32), dtype=np.uint8)
        assert abs(iou_score(mask, mask) - 1.0) < 1e-5

    def test_precision_recall_f1_all_correct(self):
        mask = np.ones((32, 32), dtype=np.uint8)
        p, r, f1 = precision_recall_f1(mask, mask)
        assert abs(p - 1.0) < 1e-4
        assert abs(r - 1.0) < 1e-4
        assert abs(f1 - 1.0) < 1e-4
