"""Tests for trainer checkpoint round-tripping and directory layout."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from histocoreml.config import TrainingConfig

pytest.importorskip("segmentation_models_pytorch")

from histocoreml.training.trainer import SegmentationTrainer  # noqa: E402


@pytest.fixture
def trainer(tmp_path: Path) -> SegmentationTrainer:
    cfg = TrainingConfig(
        architecture="unet",
        encoder="resnet18",
        pretrained=False,
        epochs=1,
        checkpoint_dir=tmp_path / "run" / "checkpoints",
        device="cpu",
        mixed_precision=False,
    )
    return SegmentationTrainer(cfg)


class TestCheckpointDirectory:
    def test_checkpoint_dir_is_not_nested_twice(self, trainer: SegmentationTrainer, tmp_path: Path):
        assert trainer.checkpoint_dir == tmp_path / "run" / "checkpoints"
        assert not (trainer.checkpoint_dir / "checkpoints").exists()


class TestSaveAndLoadCheckpoint:
    def test_round_trip_restores_state(self, trainer: SegmentationTrainer, tmp_path: Path):
        trainer.state = trainer.state.replace(epoch=4, best_metric=0.42)
        path = tmp_path / "nested" / "ckpt.pth"

        trainer.save_checkpoint(path, {"val_dice": 0.42})

        assert path.exists()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["epoch"] == 4
        assert payload["val_dice"] == pytest.approx(0.42)

        trainer.state = trainer.state.replace(epoch=0, best_metric=0.0)
        trainer.load_checkpoint(path)
        assert trainer.state.epoch == 4
        assert trainer.state.best_metric == pytest.approx(0.42)

    def test_weights_are_restored(self, trainer: SegmentationTrainer, tmp_path: Path):
        path = tmp_path / "weights.pth"
        trainer.save_checkpoint(path)

        original = next(iter(trainer.model.parameters())).clone()
        with torch.no_grad():
            next(iter(trainer.model.parameters())).add_(1.0)

        trainer.load_checkpoint(path)
        torch.testing.assert_close(next(iter(trainer.model.parameters())), original)

    def test_explicit_criterion_is_used(self, tmp_path: Path):
        from histocoreml.training.losses import get_loss

        criterion = get_loss("dice_bce", dice_weight=0.9, bce_weight=0.1)
        cfg = TrainingConfig(
            architecture="unet",
            encoder="resnet18",
            pretrained=False,
            checkpoint_dir=tmp_path / "checkpoints",
            device="cpu",
        )
        assert SegmentationTrainer(cfg, criterion=criterion).criterion is criterion
