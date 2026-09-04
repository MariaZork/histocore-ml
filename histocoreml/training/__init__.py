"""HistoCoreML training — loss functions, datasets, and training utilities.

This module provides a complete training infrastructure for histology ML models
with a consistent API based on abstract base classes.

Base Classes::

    from histocoreml.training.base_trainer import (
        BaseTrainer,           # Abstract base for all trainers
        BaseSegmentationTrainer,  # Base for segmentation trainers
    )
    from histocoreml.config import (
        TrainerConfig,         # Base configuration
        TrainingState,         # Training state tracking
    )

Datasets::

    from histocoreml.training import RLEMaskProvider, SegmentationDataset

    # Tiles slides on the fly — nothing written to disk
    dataset = SegmentationDataset(
        slide_dir=Path("data/train"),
        mask_provider=RLEMaskProvider.from_csv(Path("train.csv")),
        tiling_cfg=TilingConfig(overlap=256),
    )

    # Or read tiles already extracted to disk
    from histocoreml.training import build_train_dataloader
    loader = build_train_dataloader(Path("patches/images"), Path("patches/masks"))

Segmentation Training::

    from histocoreml.training import SegmentationTrainer, TrainingConfig

    cfg = TrainingConfig(architecture="unet", encoder="resnet50", epochs=100)
    with SegmentationTrainer(cfg) as trainer:
        history = trainer.fit(train_loader, val_loader)

Loss Functions::

    from histocoreml.training import get_loss, DiceLoss, DiceBCELoss, FocalLoss

    criterion = get_loss("dice_bce")  # or "dice", "focal", "tversky", "bce"

Metrics::

    from histocoreml.training import dice_score, iou_score, precision_recall_f1

    dice = dice_score(pred_mask, true_mask)
    iou = iou_score(pred_mask, true_mask)
"""

from histocoreml.config import (
    TrainerConfig,
    TrainingState,
)
from histocoreml.training.base_trainer import (
    BaseSegmentationTrainer,
    BaseTrainer,
)
from histocoreml.training.dataset import (
    HistoSegDataset,
    MaskProvider,
    PatchDirectoryDataset,
    RLEMaskProvider,
    SegmentationDataset,
    build_train_dataloader,
)
from histocoreml.training.losses import (
    DiceBCELoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    get_loss,
)
from histocoreml.training.metrics import (
    dice_score,
    hausdorff_distance,
    iou_score,
    precision_recall_f1,
)
from histocoreml.training.trainer import SegmentationTrainer
from histocoreml.training.transforms import build_augmentation_pair, build_transforms

__all__ = [
    # Base classes
    "BaseTrainer",
    "BaseSegmentationTrainer",
    "TrainerConfig",
    "TrainingState",
    # Segmentation trainer
    "SegmentationTrainer",
    # Datasets
    "PatchDirectoryDataset",
    "HistoSegDataset",  # deprecated alias
    "build_train_dataloader",
    "SegmentationDataset",
    "MaskProvider",
    "RLEMaskProvider",
    # Augmentation
    "build_transforms",
    "build_augmentation_pair",
    # Losses
    "DiceLoss",
    "DiceBCELoss",
    "FocalLoss",
    "TverskyLoss",
    "get_loss",
    # Metrics
    "dice_score",
    "iou_score",
    "precision_recall_f1",
    "hausdorff_distance",
]
