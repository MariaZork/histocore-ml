"""Tests for inference backend selection and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from histocoreml.backends.checkpoint_model import CheckpointModel
from histocoreml.backends.factory import get_inference_model
from histocoreml.config import ModelConfig


def _cfg(path: Path, **kwargs) -> ModelConfig:
    return ModelConfig(model_path=path, device="cpu", **kwargs)


class TestBackendSelection:
    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            (".pt", "TorchScriptModel"),
            (".ts", "TorchScriptModel"),
            (".onnx", "ONNXModel"),
            (".pth", "CheckpointModel"),
            (".ckpt", "CheckpointModel"),
        ],
    )
    def test_auto_backend_from_suffix(self, suffix: str, expected: str):
        model = get_inference_model(_cfg(Path(f"model{suffix}")))
        assert type(model).__name__ == expected

    def test_explicit_backend_overrides_suffix(self):
        model = get_inference_model(_cfg(Path("model.pth"), backend="torchscript"))
        assert type(model).__name__ == "TorchScriptModel"

    def test_unknown_suffix_raises(self):
        with pytest.raises(ValueError, match="Cannot infer an inference backend"):
            get_inference_model(_cfg(Path("model.bin")))

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown inference backend"):
            get_inference_model(_cfg(Path("model.pt"), backend="tensorrt"))


class TestCheckpointModel:
    @pytest.fixture
    def checkpoint(self, tmp_path: Path) -> Path:
        smp = pytest.importorskip("segmentation_models_pytorch")
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
        path = tmp_path / "best.pth"
        torch.save({"model_state_dict": model.state_dict()}, path)
        return path

    def test_predicts_binary_masks(self, checkpoint: Path):
        cfg = _cfg(checkpoint, architecture="unet", encoder="resnet18", patch_size=64)
        with CheckpointModel(cfg) as model:
            masks = model.predict_batch(torch.rand(2, 3, 64, 64))

        assert masks.shape == (2, 64, 64)
        assert set(masks.ravel().tolist()) <= {0, 1}

    def test_predicts_probabilities(self, checkpoint: Path):
        cfg = _cfg(checkpoint, architecture="unet", encoder="resnet18", patch_size=64)
        with CheckpointModel(cfg) as model:
            probs = model.predict_proba_batch(torch.rand(1, 3, 64, 64))

        assert probs.shape == (1, 64, 64)
        assert 0.0 <= probs.min() and probs.max() <= 1.0

    def test_architecture_read_from_checkpoint_config(self, tmp_path: Path):
        smp = pytest.importorskip("segmentation_models_pytorch")
        from histocoreml.config import TrainingConfig

        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
        path = tmp_path / "with_config.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": TrainingConfig(architecture="unet", encoder="resnet18"),
            },
            path,
        )

        # No architecture/encoder on the ModelConfig — they come from the checkpoint.
        with CheckpointModel(_cfg(path, patch_size=64)) as loaded:
            assert loaded.predict_batch(torch.rand(1, 3, 64, 64)).shape == (1, 64, 64)

    def test_missing_architecture_raises(self, tmp_path: Path):
        pytest.importorskip("segmentation_models_pytorch")
        path = tmp_path / "bare.pth"
        torch.save({"model_state_dict": {}}, path)

        with pytest.raises(ValueError, match="does not record its architecture"):
            CheckpointModel(_cfg(path)).load()

    def test_missing_file_raises(self, tmp_path: Path):
        cfg = _cfg(tmp_path / "nope.pth", architecture="unet", encoder="resnet18")
        with pytest.raises(FileNotFoundError):
            CheckpointModel(cfg).load()

    def test_predict_before_load_raises(self, tmp_path: Path):
        model = CheckpointModel(_cfg(tmp_path / "x.pth", architecture="unet", encoder="resnet18"))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict_batch(torch.rand(1, 3, 64, 64))
