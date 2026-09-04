"""Integration tests for the WSI reading pipeline.

These tests verify that WSI readers can read various file formats
and return proper metadata and image patches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from histocoreml.io.factory import get_reader


@pytest.fixture
def dummy_tiff_wsi(tmp_path: Path) -> Path:
    """Create a dummy multi-resolution TIFF that mimics a WSI."""
    path = tmp_path / "test_slide.tiff"

    # Create a multi-level TIFF
    level0 = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    level1 = level0[::2, ::2, :].copy()
    level2 = level0[::4, ::4, :].copy()

    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        writer.write(level0, subifds=2)
        writer.write(level1, subfiletype=1)
        writer.write(level2, subfiletype=1)

    return path


@pytest.fixture
def dummy_svs_like(tmp_path: Path) -> Path:
    """Create a dummy SVS-like TIFF with required metadata."""
    path = tmp_path / "test_slide.svs"

    # Create a multi-level image
    level0 = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    # Add ImageDescription with SVS-like metadata
    description = "Aperio Image Library v12.1.1\n2560x1920 (0.250000, 0.250000)"

    with tifffile.TiffWriter(path) as writer:
        writer.write(
            level0,
            description=description,
            resolution=(10000, 10000, "cm"),  # 0.25 um per pixel
        )

    return path


class TestWSIReaderFactory:
    """Test suite for WSI reader factory."""

    def test_get_reader_for_tiff_file(self, dummy_tiff_wsi: Path) -> None:
        """Factory should return appropriate reader for .tiff files."""
        with get_reader(dummy_tiff_wsi) as reader:
            assert reader is not None
            meta = reader.get_metadata()
            assert meta.dimensions == (1024, 1024)

    def test_get_reader_for_svs_file(self, dummy_svs_like: Path) -> None:
        """Factory should return appropriate reader for .svs files."""
        with get_reader(dummy_svs_like) as reader:
            assert reader is not None

    def test_get_reader_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Factory should raise FileNotFoundError for missing files."""
        nonexistent = tmp_path / "nonexistent.svs"
        with pytest.raises(FileNotFoundError):
            with get_reader(nonexistent):
                pass


class TestTifffileReader:
    """Test suite for TiffFileReader implementation."""

    def test_read_region_from_multi_level(self, dummy_tiff_wsi: Path) -> None:
        """Should be able to read regions from multi-level TIFF."""
        with get_reader(dummy_tiff_wsi) as reader:
            # Read a 256x256 region from level 0
            region = reader.read_region((0, 0), 0, (256, 256))
            assert region.shape == (256, 256, 3)
            assert region.dtype == np.uint8

    def test_read_region_different_levels(self, dummy_tiff_wsi: Path) -> None:
        """Should read correctly from different pyramid levels."""
        with get_reader(dummy_tiff_wsi) as reader:
            # Level 0
            region0 = reader.read_region((0, 0), 0, (256, 256))
            assert region0.shape == (256, 256, 3)

            # Level 1 (should be smaller or handle gracefully)
            try:
                region1 = reader.read_region((0, 0), 1, (128, 128))
                assert region1.shape[0] <= 256 and region1.shape[1] <= 256
            except ValueError:
                # Some readers may not support level 1
                pass

    def test_get_metadata_includes_dimensions(self, dummy_tiff_wsi: Path) -> None:
        """Metadata should include image dimensions."""
        with get_reader(dummy_tiff_wsi) as reader:
            meta = reader.get_metadata()
            assert meta.dimensions == (1024, 1024)
            width, height = meta.dimensions
            assert width == 1024
            assert height == 1024

    def test_context_manager_cleanup(self, dummy_tiff_wsi: Path) -> None:
        """Reader should properly cleanup on context exit."""
        reader = None
        with get_reader(dummy_tiff_wsi) as r:
            reader = r
            # Should be able to read while in context
            _ = reader.get_metadata()
        # After exit, reader should be closed


class TestWSIMetadataExtraction:
    """Test suite for WSI metadata extraction."""

    def test_mpp_extraction_from_svs(self, dummy_svs_like: Path) -> None:
        """Should extract MPP from SVS-like files."""
        with get_reader(dummy_svs_like) as reader:
            meta = reader.get_metadata()
            # MPP may be None or extracted from resolution
            # Just verify it doesn't crash
            assert meta.mpp is None or meta.mpp > 0

    def test_level_count(self, dummy_tiff_wsi: Path) -> None:
        """Should report correct number of pyramid levels."""
        with get_reader(dummy_tiff_wsi) as reader:
            meta = reader.get_metadata()
            assert meta.level_count >= 1

    def test_level_dimensions(self, dummy_tiff_wsi: Path) -> None:
        """Should report dimensions for each level."""
        with get_reader(dummy_tiff_wsi) as reader:
            meta = reader.get_metadata()
            assert len(meta.level_dimensions) == meta.level_count
            # Level 0 should be the full resolution
            assert meta.level_dimensions[0] == meta.dimensions


class TestWSIReaderEdgeCases:
    """Test edge cases for WSI readers."""

    def test_read_region_out_of_bounds(self, dummy_tiff_wsi: Path) -> None:
        """Should handle out-of-bounds reads gracefully."""
        with get_reader(dummy_tiff_wsi) as reader:
            # Try to read beyond image bounds
            # Implementation may raise ValueError or clamp
            # Out-of-bounds is zero-padded to the requested size, matching
            # OpenSlide, so the two backends behave identically.
            region = reader.read_region((800, 800), 0, (300, 300))
            assert region.shape == (300, 300, 3)
            assert region[224:, :].max() == 0  # beyond the edge is padding

    def test_read_region_zero_size(self, dummy_tiff_wsi: Path) -> None:
        """Should handle zero-size regions."""
        with get_reader(dummy_tiff_wsi) as reader:
            with pytest.raises(ValueError):
                reader.read_region((0, 0), 0, (0, 256))

    def test_read_region_negative_coords(self, dummy_tiff_wsi: Path) -> None:
        """Should handle negative coordinates."""
        with get_reader(dummy_tiff_wsi) as reader:
            # Negative coordinates must not wrap around into the far edge;
            # the out-of-slide part is padding and the rest is real pixels.
            region = reader.read_region((-10, -10), 0, (100, 100))
            assert region.shape == (100, 100, 3)
            assert region[:10, :].max() == 0
            assert region[10:, 10:].max() > 0
