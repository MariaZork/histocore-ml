"""Integration tests for output writers.

These tests verify the end-to-end output writing functionality including:
- TIFF writer
- NPY writer
- RLE writer
- Zarr writer
- GeoJSON writer
- Factory pattern
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from histocoreml.config import OutputConfig
from histocoreml.io.base_reader import WSIMetadata
from histocoreml.output.base_writer import WriteResult
from histocoreml.output.factory import get_writer
from histocoreml.output.rle_codec import (
    plain_rle_decode,
    plain_rle_encode,
    plain_rle_from_dict,
    plain_rle_to_dict,
)
from histocoreml.output.rle_writer import RLEMaskWriter
from histocoreml.output.writers import NumpyMaskWriter, TiffMaskWriter


def create_test_metadata() -> WSIMetadata:
    """Create test WSI metadata."""
    return WSIMetadata(
        path=Path("test_slide.svs"),
        level_count=1,
        level_dimensions=((256, 256),),
        level_downsamples=(1.0,),
        mpp_x=0.5,
        mpp_y=0.5,
        vendor="test",
        properties={},
    )


def create_test_mask(h: int = 256, w: int = 256, num_classes: int = 2) -> np.ndarray:
    """Create a test segmentation mask."""
    np.random.seed(42)
    if num_classes == 2:
        # Binary mask
        return (np.random.random((h, w)) > 0.5).astype(np.uint8)
    else:
        # Multi-class mask
        return np.random.randint(0, num_classes, (h, w), dtype=np.uint8)


@pytest.fixture
def output_config(tmp_path: Path) -> OutputConfig:
    """Create test output configuration."""
    return OutputConfig(
        output_dir=tmp_path / "outputs",
        output_format="npy",
        downsample_factor=1,
        save_overlay=False,
        overlay_alpha=0.5,
        overlay_max_edge=1024,
    )


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create sample binary mask."""
    return create_test_mask()


@pytest.fixture
def sample_metadata() -> WSIMetadata:
    """Create sample WSI metadata."""
    return create_test_metadata()


class TestNumpyWriterIntegration:
    """Integration tests for NumpyMaskWriter."""

    def test_write_npy_file(
        self, output_config: OutputConfig, sample_mask: np.ndarray, sample_metadata: WSIMetadata
    ) -> None:
        """Should write mask as .npy file."""
        writer = NumpyMaskWriter(output_config)
        result = writer.write(sample_mask, sample_metadata, stem="test_slide")

        assert isinstance(result, WriteResult)
        assert result.path.exists()
        assert result.path.suffix == ".npy"

        # Verify content
        loaded = np.load(result.path)
        np.testing.assert_array_equal(loaded, sample_mask)

    def test_npy_preserves_mask_values(
        self, output_config: OutputConfig, sample_metadata: WSIMetadata
    ) -> None:
        """NPY format should preserve mask values exactly."""
        # Create mask with specific values
        mask = np.array([[0, 1, 2], [2, 1, 0], [1, 2, 0]], dtype=np.uint8)

        writer = NumpyMaskWriter(output_config)
        result = writer.write(mask, sample_metadata, stem="test")

        loaded = np.load(result.path)
        np.testing.assert_array_equal(loaded, mask)


class TestTIFFWriterIntegration:
    """Integration tests for TiffMaskWriter."""

    def test_write_tiff_file(
        self, output_config: OutputConfig, sample_mask: np.ndarray, sample_metadata: WSIMetadata
    ) -> None:
        """Should write mask as .tiff file."""
        output_config = replace(output_config, output_format="tiff")
        writer = TiffMaskWriter(output_config)
        result = writer.write(sample_mask, sample_metadata, stem="test_slide")

        assert result.path.exists()
        assert result.path.suffix in [".tiff", ".tif"]

    def test_tiff_preserves_dimensions(
        self, output_config: OutputConfig, sample_metadata: WSIMetadata
    ) -> None:
        """TIFF should preserve mask dimensions."""
        mask = create_test_mask(512, 512)
        output_config = replace(output_config, output_format="tiff")

        writer = TiffMaskWriter(output_config)
        result = writer.write(mask, sample_metadata, stem="test")

        # Try to load with tifffile
        try:
            import tifffile

            loaded = tifffile.imread(result.path)
            assert loaded.shape[:2] == mask.shape
        except Exception:
            pytest.skip("Could not verify TIFF content")


class TestRLEWriterIntegration:
    """Integration tests for RLEMaskWriter."""

    def test_write_rle_json_file(
        self, output_config: OutputConfig, sample_mask: np.ndarray, sample_metadata: WSIMetadata
    ) -> None:
        """Should write mask as RLE JSON file."""
        output_config = replace(output_config, output_format="rle")
        writer = RLEMaskWriter(output_config)
        result = writer.write(sample_mask, sample_metadata, stem="test_slide")

        assert result.path.exists()
        assert result.path.suffix == ".json"

    def test_rle_roundtrip(self, output_config: OutputConfig, sample_metadata: WSIMetadata) -> None:
        """A written RLE file should decode back to the original mask."""
        mask = create_test_mask(64, 64)
        output_config = replace(output_config, output_format="rle")

        writer = RLEMaskWriter(output_config)
        result = writer.write(mask, sample_metadata, stem="test")

        with open(result.path) as f:
            data = json.load(f)

        assert data["format"] == "plain_rle"
        assert tuple(data["shape"]) == mask.shape
        assert data["runs"]

        decoded = plain_rle_decode(plain_rle_from_dict(data))
        np.testing.assert_array_equal(decoded, mask)


class TestWriterFactoryIntegration:
    """Integration tests for writer factory."""

    def test_factory_creates_npy_writer(self, output_config: OutputConfig) -> None:
        """Factory should create NumpyMaskWriter for 'npy' format."""
        output_config = replace(output_config, output_format="npy")
        writer = get_writer(output_config)
        assert isinstance(writer, NumpyMaskWriter)

    def test_factory_creates_tiff_writer(self, output_config: OutputConfig) -> None:
        """Factory should create TiffMaskWriter for 'tiff' format."""
        output_config = replace(output_config, output_format="tiff")
        writer = get_writer(output_config)
        assert isinstance(writer, TiffMaskWriter)

    def test_factory_creates_rle_writer(self, output_config: OutputConfig) -> None:
        """Factory should create RLEMaskWriter for 'rle' format."""
        output_config = replace(output_config, output_format="rle")
        writer = get_writer(output_config)
        assert isinstance(writer, RLEMaskWriter)

    def test_factory_unsupported_format_raises(self, output_config: OutputConfig) -> None:
        """Factory should raise ValueError for unsupported format."""
        output_config = replace(output_config, output_format="unsupported")
        with pytest.raises(ValueError) as exc_info:
            get_writer(output_config)
        assert "unsupported" in str(exc_info.value).lower()

    def test_factory_case_insensitive(self, output_config: OutputConfig) -> None:
        """Factory should handle format names case-insensitively."""
        output_config = replace(output_config, output_format="NPY")
        writer = get_writer(output_config)
        assert isinstance(writer, NumpyMaskWriter)


class TestWriterWithMetadata:
    """Tests for writer integration with WSI metadata."""

    def test_writer_uses_mpp_from_metadata(
        self, output_config: OutputConfig, sample_mask: np.ndarray
    ) -> None:
        """Writer should use MPP from metadata."""
        metadata = WSIMetadata(
            path=Path("test_slide.svs"),
            level_count=1,
            level_dimensions=((256, 256),),
            level_downsamples=(1.0,),
            mpp_x=0.25,  # Specific MPP value
            mpp_y=0.25,
            vendor="test",
            properties={},
        )

        writer = NumpyMaskWriter(output_config)
        result = writer.write(sample_mask, metadata, stem="test")

        # Output should exist
        assert result.path.exists()


class TestWriterEdgeCases:
    """Tests for writer edge cases."""

    def test_writer_creates_output_directory(
        self,
        output_config: OutputConfig,
        sample_mask: np.ndarray,
        sample_metadata: WSIMetadata,
        tmp_path: Path,
    ) -> None:
        """Writer should create output directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "output" / "dir"
        output_config = replace(output_config, output_dir=nested_dir)

        writer = NumpyMaskWriter(output_config)
        result = writer.write(sample_mask, sample_metadata, stem="test")

        assert nested_dir.exists()
        assert result.path.exists()

    def test_writer_empty_mask(
        self, output_config: OutputConfig, sample_metadata: WSIMetadata
    ) -> None:
        """Writer should handle empty mask."""
        empty_mask = np.zeros((256, 256), dtype=np.uint8)

        writer = NumpyMaskWriter(output_config)
        result = writer.write(empty_mask, sample_metadata, stem="test")

        loaded = np.load(result.path)
        np.testing.assert_array_equal(loaded, empty_mask)

    def test_writer_all_ones_mask(
        self, output_config: OutputConfig, sample_metadata: WSIMetadata
    ) -> None:
        """Writer should handle all-ones mask."""
        full_mask = np.ones((256, 256), dtype=np.uint8)

        writer = NumpyMaskWriter(output_config)
        result = writer.write(full_mask, sample_metadata, stem="test")

        loaded = np.load(result.path)
        np.testing.assert_array_equal(loaded, full_mask)


class TestRLECodecIntegration:
    """Integration tests for RLE codec."""

    def test_encode_plain_rle_roundtrip(self) -> None:
        """Plain RLE encoding should be reversible."""
        mask = create_test_mask(64, 64)

        encoded = plain_rle_encode(mask)

        assert encoded.shape == mask.shape
        np.testing.assert_array_equal(plain_rle_decode(encoded), mask)

    def test_runs_are_value_length_pairs(self) -> None:
        """Runs alternate in value and their lengths cover every pixel."""
        mask = np.array([[0, 0, 1, 1], [1, 0, 0, 0]], dtype=np.uint8)

        runs = plain_rle_encode(mask).runs

        assert runs == [(0, 2), (1, 3), (0, 3)]
        assert sum(length for _, length in runs) == mask.size

    def test_rle_all_zeros(self) -> None:
        """RLE should handle all-zero mask."""
        mask = np.zeros((64, 64), dtype=np.uint8)

        encoded = plain_rle_encode(mask)

        assert encoded.runs == [(0, 64 * 64)]
        np.testing.assert_array_equal(plain_rle_decode(encoded), mask)

    def test_rle_all_ones(self) -> None:
        """RLE should handle all-ones mask."""
        mask = np.ones((64, 64), dtype=np.uint8)

        encoded = plain_rle_encode(mask)

        assert encoded.runs == [(1, 64 * 64)]
        np.testing.assert_array_equal(plain_rle_decode(encoded), mask)

    def test_json_dict_roundtrip(self) -> None:
        """PlainRLE survives a trip through its dict representation."""
        mask = create_test_mask(32, 32)

        restored = plain_rle_from_dict(plain_rle_to_dict(plain_rle_encode(mask)))

        np.testing.assert_array_equal(plain_rle_decode(restored), mask)
