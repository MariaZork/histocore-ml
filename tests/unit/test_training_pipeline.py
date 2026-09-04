"""Tests for the on-the-fly WSI segmentation training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

from histocoreml.config import ExperimentConfig
from histocoreml.pipelines.training.segmentation import SegmentationTrainingPipeline
from histocoreml.training.dataset import RLEMaskProvider

pytest.importorskip("segmentation_models_pytorch")


def _rle(mask: np.ndarray) -> str:
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A miniature two-slide dataset shaped like HuBMAP's."""
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()

    rng = np.random.default_rng(0)
    rows = ["id,encoding"]
    for i in range(2):
        image = np.full((512, 512, 3), 245, dtype=np.uint8)
        image[64:448, 64:448] = rng.integers(60, 190, (384, 384, 3), dtype=np.uint8)
        tifffile.imwrite(root / "train" / f"slide_{i}.tiff", image)

        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[128:256, 128:256] = 1
        rows.append(f"slide_{i},{_rle(mask)}")

    (root / "train.csv").write_text("\n".join(rows) + "\n")

    test_image = np.full((512, 512, 3), 245, dtype=np.uint8)
    test_image[64:448, 64:448] = rng.integers(60, 190, (384, 384, 3), dtype=np.uint8)
    tifffile.imwrite(root / "test" / "test_0.tiff", test_image)
    return root


@pytest.fixture
def config(tmp_path: Path, dataset_root: Path) -> ExperimentConfig:
    document = {
        "experiment": {"name": "test_run", "output_dir": str(tmp_path / "out"), "seed": 0},
        "data": {
            "train_dir": str(dataset_root / "train"),
            "train_csv": str(dataset_root / "train.csv"),
            "test_dir": str(dataset_root / "test"),
            "patch_size": 128,
            "patch_overlap": 0.0,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "val_split": 0.5,
            "tissue_threshold": 0.5,
            "augmentation": {"enabled": False},
        },
        "model": {"name": "unet", "encoder_name": "resnet18", "encoder_pretrained": False},
        "training": {
            "epochs": 1,
            "optimizer": {"name": "AdamW", "lr": 1e-3},
            "loss": {"name": "dice_bce", "dice_weight": 0.5, "bce_weight": 0.5},
            "mixed_precision": False,
            "debug_samples": 4,
            "viz_interval": 0,
        },
        "inference": {"patch_size": 128, "batch_size": 2, "threshold": 0.5, "output_format": "npy"},
    }
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(document))
    cfg = ExperimentConfig.from_yaml(path)
    cfg.device = "cpu"
    return cfg


class TestSegmentationTrainingPipeline:
    def test_train_and_infer(self, config: ExperimentConfig):
        pipeline = SegmentationTrainingPipeline(config, debug=True)
        result = pipeline.run()

        assert result.success, result.errors
        assert result.epochs_trained == 1
        assert result.checkpoint_path is not None and result.checkpoint_path.exists()
        assert "train_loss" in result.history
        assert result.metadata["inference"] == {"total": 1, "successful": 1}
        assert (config.output_dir / "test_predictions").exists()

    def test_checkpoints_are_not_double_nested(self, config: ExperimentConfig):
        SegmentationTrainingPipeline(config, debug=True).run(infer=False)

        checkpoints = config.output_dir / "checkpoints"
        assert (checkpoints / "best.pth").exists()
        assert not (checkpoints / "checkpoints").exists()

    def test_inference_only_skips_training(self, config: ExperimentConfig):
        SegmentationTrainingPipeline(config, debug=True).run(infer=False)

        result = SegmentationTrainingPipeline(config).run(train=False)

        assert result.success, result.errors
        assert result.epochs_trained == 0
        assert result.metadata["inference"]["successful"] == 1

    def test_split_is_by_slide(self, config: ExperimentConfig):
        pipeline = SegmentationTrainingPipeline(config)
        train_ids, val_ids = pipeline.split_slide_ids()

        assert set(train_ids).isdisjoint(val_ids)
        assert sorted(train_ids + val_ids) == ["slide_0", "slide_1"]

    def test_errors_are_captured_not_raised(self, config: ExperimentConfig):
        # An empty provider yields no patches, which the pipeline reports as an error.
        pipeline = SegmentationTrainingPipeline(
            config, mask_provider=RLEMaskProvider({"absent_slide": "1 1"})
        )
        result = pipeline.run(infer=False)

        assert not result.success
        assert any("No training patches" in e for e in result.errors)
