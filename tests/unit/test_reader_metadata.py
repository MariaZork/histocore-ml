"""Tests for WSI metadata resolution and TIFF axis handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from histocoreml.io.base_reader import WSIMetadata
from histocoreml.io.factory import get_reader


def _metadata(mpp: float | None) -> WSIMetadata:
    return WSIMetadata(
        path=Path("slide.tiff"),
        level_count=2,
        level_dimensions=((1024, 512), (512, 256)),
        level_downsamples=(1.0, 2.0),
        mpp_x=mpp,
        mpp_y=mpp,
        vendor=None,
        properties={},
    )


class TestLevelForMPP:
    def test_matches_best_level_when_mpp_known(self):
        metadata = _metadata(0.5)
        assert metadata.level_for_mpp(1.0) == metadata.best_level_for_mpp(1.0)

    def test_falls_back_to_level_zero_without_mpp(self):
        metadata = _metadata(None)

        assert metadata.level_for_mpp(0.5) == (0, 0.5)
        with pytest.raises(ValueError, match="mpp metadata missing"):
            metadata.best_level_for_mpp(0.5)


class TestTiffAxisHandling:
    @pytest.mark.parametrize(
        ("shape", "expected_dimensions"),
        [
            ((512, 400, 3), (400, 512)),  # channel-last RGB
            ((512, 400), (400, 512)),  # grayscale, no channel axis
        ],
    )
    def test_dimensions_are_width_height(
        self, tmp_path: Path, shape: tuple[int, ...], expected_dimensions: tuple[int, int]
    ):
        path = tmp_path / "slide.tiff"
        tifffile.imwrite(path, np.zeros(shape, dtype=np.uint8))

        with get_reader(path) as reader:
            assert reader.get_metadata().dimensions == expected_dimensions

    def test_regions_and_thumbnails_are_rgb(self, tmp_path: Path):
        path = tmp_path / "gray.tiff"
        tifffile.imwrite(path, np.full((256, 256), 120, dtype=np.uint8))

        with get_reader(path) as reader:
            assert reader.read_region((0, 0), 0, (64, 64)).shape == (64, 64, 3)
            assert reader.get_thumbnail().shape[-1] == 3

    def test_region_content_matches_source(self, tmp_path: Path):
        source = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        path = tmp_path / "rgb.tiff"
        tifffile.imwrite(path, source)

        with get_reader(path) as reader:
            region = reader.read_region((32, 16), 0, (64, 48))

        np.testing.assert_array_equal(region, source[16:64, 32:96])
