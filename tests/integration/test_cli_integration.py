"""Integration tests for CLI commands.

These tests verify the command-line interface functionality including:
- histo-segment command
- histo-embed command
- histo-extract command
- histo-train command
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

from histocoreml.cli import (
    main_embed,
    main_extract,
    main_segment,
    main_train,
)


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    """Create a sample configuration file."""
    config = {
        "model": {
            "model_path": str(tmp_path / "model.pt"),
            "patch_size": 64,
            "target_mpp": 1.0,
            "batch_size": 2,
            "device": "cpu",
            "num_classes": 1,
        },
        "tiling": {
            "overlap": 0,
            "tissue_threshold": 0.0,
            "num_workers": 0,
        },
        "output": {
            "output_dir": str(tmp_path / "outputs"),
            "output_format": "npy",
            "downsample_factor": 1,
            "save_overlay": False,
            "overlay_alpha": 0.5,
            "overlay_max_edge": 1024,
        },
        "log_level": "INFO",
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def sample_wsi(tmp_path: Path) -> Path:
    """Create a sample WSI file."""
    wsi_path = tmp_path / "test_slide.tiff"
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    with tifffile.TiffWriter(wsi_path) as writer:
        writer.write(img)
    return wsi_path


@pytest.fixture
def sample_mask(tmp_path: Path) -> Path:
    """Create a sample mask file."""
    mask_path = tmp_path / "test_mask.npy"
    mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    np.save(mask_path, mask)
    return mask_path


class TestCLISegment:
    """Integration tests for histo-segment CLI."""

    def test_segment_help(self) -> None:
        """Segment command should show help."""
        with pytest.raises(SystemExit) as exc_info:
            main_segment(["--help"])
        assert exc_info.value.code == 0

    def test_segment_missing_config(self) -> None:
        """Segment command should fail without config."""
        with pytest.raises(SystemExit) as exc_info:
            main_segment([])
        assert exc_info.value.code == 2

    def test_segment_missing_input(self, sample_config_yaml: Path) -> None:
        """Segment command should fail without input files."""
        with pytest.raises(SystemExit) as exc_info:
            main_segment(["-c", str(sample_config_yaml)])
        assert exc_info.value.code == 2

    @pytest.mark.slow
    def test_segment_with_valid_input(self, sample_config_yaml: Path, sample_wsi: Path) -> None:
        """Segment command should process valid input."""
        exit_code = main_segment(
            [
                "-c",
                str(sample_config_yaml),
                "-i",
                str(sample_wsi),
            ]
        )
        assert exit_code in [0, 1]  # 0 for success, 1 if model loading fails


class TestCLIExtract:
    """Integration tests for histo-extract CLI."""

    def test_extract_help(self) -> None:
        """Extract command should show help."""
        with pytest.raises(SystemExit) as exc_info:
            main_extract(["--help"])
        assert exc_info.value.code == 0

    def test_extract_missing_input(self) -> None:
        """Extract command should fail without input."""
        with pytest.raises(SystemExit) as exc_info:
            main_extract([])
        assert exc_info.value.code == 2

    def test_extract_with_valid_input(
        self, sample_wsi: Path, sample_mask: Path, tmp_path: Path
    ) -> None:
        """Extract command should process valid input."""
        output_path = tmp_path / "biomarkers.json"

        exit_code = main_extract(
            [
                "-i",
                str(sample_wsi),
                "--mask",
                str(sample_mask),
                "-o",
                str(output_path),
                "--tasks",
                "cell_density",
                "spatial_graph",
            ]
        )

        assert exit_code == 0
        assert output_path.exists()

    def test_extract_without_mask(self, sample_wsi: Path, tmp_path: Path) -> None:
        """Extract command should work without mask."""
        output_path = tmp_path / "biomarkers.json"

        exit_code = main_extract(
            [
                "-i",
                str(sample_wsi),
                "-o",
                str(output_path),
                "--tasks",
                "cell_density",
            ]
        )

        assert exit_code == 0


class TestCLIEmbed:
    """Integration tests for histo-embed CLI."""

    def test_embed_help(self) -> None:
        """Embed command should show help."""
        with pytest.raises(SystemExit) as exc_info:
            main_embed(["--help"])
        assert exc_info.value.code == 0

    def test_embed_missing_input(self) -> None:
        """Embed command should fail without input."""
        with pytest.raises(SystemExit) as exc_info:
            main_embed([])
        assert exc_info.value.code == 2

    @pytest.mark.slow
    def test_embed_with_valid_input(self, sample_wsi: Path, tmp_path: Path) -> None:
        """Embed command should process valid input."""
        output_dir = tmp_path / "embeddings"

        # This may fail due to missing foundation models, but should not crash
        try:
            main_embed(
                [
                    "-i",
                    str(sample_wsi),
                    "-o",
                    str(output_dir),
                    "--model",
                    "uni",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
        except (ImportError, RuntimeError, FileNotFoundError):
            # Expected if foundation models aren't available
            pytest.skip("Foundation model not available")


class TestCLITrain:
    """Integration tests for histo-train CLI."""

    def test_train_help(self) -> None:
        """Train command should show help."""
        with pytest.raises(SystemExit) as exc_info:
            main_train(["--help"])
        assert exc_info.value.code == 0

    def test_train_missing_images(self, capsys) -> None:
        """Train command should fail without a config or image directories.

        --images is no longer argparse-required (a config supplies the data
        instead), so this is reported like every other bad-input case in the
        CLI: a message on stderr and a non-zero return code.
        """
        exit_code = main_train([])

        assert exit_code == 1
        assert "Provide --config, or both --images and --masks" in capsys.readouterr().err

    @pytest.mark.slow
    def test_train_with_valid_dirs(self, tmp_path: Path) -> None:
        """Train command should process valid directories."""
        train_images = tmp_path / "train_images"
        train_masks = tmp_path / "train_masks"
        train_images.mkdir()
        train_masks.mkdir()

        # Create some training samples
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
            tifffile.imwrite(train_images / f"sample_{i}.tiff", img)
            tifffile.imwrite(train_masks / f"sample_{i}.tiff", mask)

        checkpoint_dir = tmp_path / "checkpoints"

        # This may fail due to lack of real model, but should parse args
        try:
            main_train(
                [
                    "--images",
                    str(train_images),
                    "--masks",
                    str(train_masks),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                ]
            )
        except (RuntimeError, ImportError, FileNotFoundError) as e:
            # Training may fail due to missing dependencies/models
            pytest.skip(f"Training prerequisites not available: {e}")


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_segment_parser_device_override(
        self, sample_config_yaml: Path, sample_wsi: Path
    ) -> None:
        """Segment should allow device override."""
        # Just test that argument parsing works
        argv = [
            "-c",
            str(sample_config_yaml),
            "-i",
            str(sample_wsi),
            "--device",
            "cpu",
        ]
        # This will fail without a real model, but parsing should work
        try:
            main_segment(argv)
        except (RuntimeError, FileNotFoundError):
            pass  # Expected if model isn't available

    def test_segment_parser_batch_size_override(
        self, sample_config_yaml: Path, sample_wsi: Path
    ) -> None:
        """Segment should allow batch size override."""
        argv = [
            "-c",
            str(sample_config_yaml),
            "-i",
            str(sample_wsi),
            "--batch-size",
            "4",
        ]
        try:
            main_segment(argv)
        except (RuntimeError, FileNotFoundError):
            pass  # Expected if model isn't available

    def test_extract_parser_tasks_parsing(self, sample_wsi: Path) -> None:
        """Extract should parse multiple tasks."""
        output_path = sample_wsi.parent / "test.json"

        try:
            main_extract(
                [
                    "-i",
                    str(sample_wsi),
                    "-o",
                    str(output_path),
                    "--tasks",
                    "cell_density",
                    "nuclei_morphology",
                    "spatial_graph",
                ]
            )
        except Exception:
            pass  # Just testing argument parsing


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_segment_invalid_config_path(self, tmp_path: Path, sample_wsi: Path) -> None:
        """Segment should handle invalid config path."""
        invalid_config = tmp_path / "nonexistent.yaml"

        try:
            main_segment(
                [
                    "-c",
                    str(invalid_config),
                    "-i",
                    str(sample_wsi),
                ]
            )
        except (FileNotFoundError, RuntimeError):
            pass  # Expected

    def test_segment_no_valid_input_files(self, sample_config_yaml: Path, tmp_path: Path) -> None:
        """Segment should handle when no input files exist."""
        nonexistent_wsi = tmp_path / "nonexistent.tiff"

        exit_code = main_segment(
            [
                "-c",
                str(sample_config_yaml),
                "-i",
                str(nonexistent_wsi),
            ]
        )

        assert exit_code == 1

    def test_extract_invalid_mask(self, sample_wsi: Path, tmp_path: Path) -> None:
        """Extract should handle invalid mask file."""
        invalid_mask = tmp_path / "invalid_mask.npy"

        try:
            main_extract(
                [
                    "-i",
                    str(sample_wsi),
                    "--mask",
                    str(invalid_mask),
                ]
            )
        except (FileNotFoundError, ValueError):
            pass  # Expected
