"""HistoCoreML training — loss functions, datasets, and training utilities.

Usage::

    from histocoreml.training import SegmentationTrainer, HistoSegDataset
    from histocoreml.config import TrainingConfig

    cfg     = TrainingConfig.from_yaml("configs/training.yaml")
    trainer = SegmentationTrainer(cfg)
    trainer.fit(train_slides, val_slides)
"""

from histocoreml.training.losses import DiceLoss, DiceBCELoss, FocalLoss, TverskyLoss
from histocoreml.training.dataset import HistoSegDataset, build_train_dataloader
from histocoreml.training.trainer import SegmentationTrainer
from histocoreml.training.metrics import dice_score, iou_score, precision_recall_f1

__all__ = [
    "DiceLoss", "DiceBCELoss", "FocalLoss", "TverskyLoss",
    "HistoSegDataset", "build_train_dataloader",
    "SegmentationTrainer",
    "dice_score", "iou_score", "precision_recall_f1",
]
