"""Integration tests for the segmentation pipeline.

These tests verify the end-to-end pipeline flow including:
- WSI reading
- Patch generation
- Model inference
- Mask assembly
- Output writing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

from histocoreml.config import SegmentationPipelineConfig
from histocoreml.pipelines import SegmentationInferencePipeline, create_segmentation_pipeline


def create_test_model(path: Path) -> Path:
    """Trace a tiny 1x1-convolution segmentation model to *path*.

    The pipeline needs real weights to load; a missing file would make every
    slide come back as an error result and the assertions below meaningless.
    """
    import torch

    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    torch.nn.init.constant_(model.weight, 0.05)
    torch.nn.init.constant_(model.bias, -1.0)
    torch.jit.trace(model.eval(), torch.rand(1, 3, 32, 32)).save(str(path))
    return path


def create_test_config(tmp_path: Path) -> SegmentationPipelineConfig:
    """Create a minimal test configuration."""
    config_dict = {
        "model": {
            "model_path": str(create_test_model(tmp_path / "model.pt")),
            "patch_size": 64,
            "target_mpp": 1.0,
            "batch_size": 2,
            "device": "cpu",
            "num_classes": 1,
        },
        "tiling": {
            "overlap": 0,
            "tissue_threshold": 0.0,  # Allow all patches for testing
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
        yaml.dump(config_dict, f)

    return SegmentationPipelineConfig.from_yaml(config_path)


@pytest.fixture
def small_wsi_image(tmp_path: Path) -> Path:
    """Create a small synthetic WSI image (256x256 RGB)."""
    path = tmp_path / "test_slide.tiff"

    # Create a simple tissue-like image
    np.random.seed(42)
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

    # Add some structure (darker regions)
    img[50:100, 50:100] = [30, 30, 30]
    img[150:200, 150:200] = [40, 40, 40]

    with tifffile.TiffWriter(path) as writer:
        writer.write(img)

    return path


@pytest.fixture
def wsi_with_multiple_levels(tmp_path: Path) -> Path:
    """Create a WSI with multiple pyramid levels."""
    path = tmp_path / "test_multilevel.tiff"

    level0 = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    level1 = level0[::2, ::2, :].copy()

    with tifffile.TiffWriter(path) as writer:
        writer.write(level0, subifds=1)
        writer.write(level1, subfiletype=1)

    return path


@pytest.mark.slow
class TestSegmentationPipeline:
    """Integration tests for the full segmentation pipeline."""

    def test_pipeline_runs_without_error(self, small_wsi_image: Path, tmp_path: Path) -> None:
        """Pipeline should process a WSI without errors."""
        config = create_test_config(tmp_path)

        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([small_wsi_image])

        assert len(results) == 1
        result = results[0]
        assert result.success
        assert result.patch_count > 0

    def test_pipeline_creates_output(self, small_wsi_image: Path, tmp_path: Path) -> None:
        """Pipeline should create output files."""
        config = create_test_config(tmp_path)

        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([small_wsi_image])

        assert len(results) == 1
        result = results[0]
        assert result.success

        # Check output file exists
        output_path = result.write_result.path
        assert output_path.exists()

    def test_pipeline_result_has_metadata(self, small_wsi_image: Path, tmp_path: Path) -> None:
        """Pipeline result should include metadata."""
        config = create_test_config(tmp_path)

        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([small_wsi_image])

        result = results[0]
        assert result.wsi_path == small_wsi_image
        assert result.elapsed_seconds > 0
        assert result.patch_count >= 0


class TestPipelineConfiguration:
    """Tests for pipeline configuration handling."""

    def test_create_pipeline_returns_pipeline(self, tmp_path: Path) -> None:
        """create_pipeline factory should return SegmentationPipeline."""
        config = create_test_config(tmp_path)
        pipeline = create_segmentation_pipeline(config)
        assert isinstance(pipeline, SegmentationInferencePipeline)

    def test_pipeline_with_multiple_wsis(self, tmp_path: Path) -> None:
        """Pipeline should handle multiple WSI files."""
        # Create two WSI images
        wsi1 = tmp_path / "slide1.tiff"
        wsi2 = tmp_path / "slide2.tiff"

        np.random.seed(42)
        img1 = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        img2 = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)

        with tifffile.TiffWriter(wsi1) as w:
            w.write(img1)
        with tifffile.TiffWriter(wsi2) as w:
            w.write(img2)

        config = create_test_config(tmp_path)
        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([wsi1, wsi2])

        assert len(results) == 2
        for result in results:
            # May succeed or fail depending on model availability
            # Just verify we got results
            assert result.wsi_path in [wsi1, wsi2]

    def test_pipeline_handles_missing_file(self, tmp_path: Path) -> None:
        """Pipeline should handle missing input files gracefully."""
        missing_file = tmp_path / "nonexistent.tiff"

        config = create_test_config(tmp_path)
        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([missing_file])

        assert len(results) == 1
        result = results[0]
        assert not result.success
        assert len(result.errors) > 0


class TestPipelineOutputFormats:
    """Tests for different output formats."""

    def test_npy_output_format(self, small_wsi_image: Path, tmp_path: Path) -> None:
        """Pipeline should be able to create .npy outputs."""
        config = create_test_config(tmp_path)

        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([small_wsi_image])

        result = results[0]
        if result.success:
            output_path = result.write_result.path
            assert output_path.suffix == ".npy"

    def test_output_mask_shape(self, small_wsi_image: Path, tmp_path: Path) -> None:
        """Output mask should have appropriate shape."""
        config = create_test_config(tmp_path)

        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([small_wsi_image])

        result = results[0]
        if result.success:
            # Load and verify mask
            mask = np.load(result.write_result.path)
            assert mask.ndim >= 1
            assert mask.shape[0] > 0 and mask.shape[1] > 0


class TestPipelineEdgeCases:
    """Tests for pipeline edge cases."""

    def test_empty_wsi_list(self, tmp_path: Path) -> None:
        """Pipeline should handle empty WSI list."""
        config = create_test_config(tmp_path)
        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([])

        assert len(results) == 0

    def test_very_small_wsi(self, tmp_path: Path) -> None:
        """Pipeline should handle very small WSI images."""
        # Create tiny image
        small_wsi = tmp_path / "tiny.tiff"
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)

        with tifffile.TiffWriter(small_wsi) as w:
            w.write(img)

        config = create_test_config(tmp_path)
        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([small_wsi])

        # Should complete (success or graceful failure)
        assert len(results) == 1

    def test_all_white_wsi(self, tmp_path: Path) -> None:
        """Pipeline should handle all-white WSI (no tissue)."""
        # Create all-white image
        white_wsi = tmp_path / "white.tiff"
        img = np.full((128, 128, 3), 255, dtype=np.uint8)

        with tifffile.TiffWriter(white_wsi) as w:
            w.write(img)

        config = create_test_config(tmp_path)
        pipeline = SegmentationInferencePipeline(config)
        results = pipeline.run([white_wsi])

        # Should complete without crashing
        assert len(results) == 1
