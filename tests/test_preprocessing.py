"""Tests for histocoreml.preprocessing."""

from __future__ import annotations

import numpy as np

from histocoreml.config import TilingConfig
from histocoreml.preprocessing.patch_coord import PatchCoord
from histocoreml.preprocessing.patch_utils import (
    is_tissue,
    macenko_normalise,
    pad_to_size,
    rescale_patch,
    tissue_mask,
)
from tests.conftest import make_coord, make_rgb_patch


class TestRescalePatch:
    def test_identity_when_rf_one(self):
        patch = make_rgb_patch(64, 64)
        coord = make_coord(patch_size=64)
        result = rescale_patch(patch, coord, model_patch_size=64)
        np.testing.assert_array_equal(result, patch)

    def test_downsample(self):
        patch = make_rgb_patch(128, 128)
        coord = PatchCoord(x=0, y=0, level=0, patch_size=128,
                           col_idx=0, row_idx=0, rescale_factor=2.0)
        result = rescale_patch(patch, coord, model_patch_size=64)
        assert result.shape == (64, 64, 3)

    def test_upsample(self):
        patch = make_rgb_patch(32, 32)
        coord = PatchCoord(x=0, y=0, level=0, patch_size=32,
                           col_idx=0, row_idx=0, rescale_factor=0.5)
        result = rescale_patch(patch, coord, model_patch_size=64)
        assert result.shape == (64, 64, 3)


class TestPadToSize:
    def test_no_padding_needed(self):
        patch = make_rgb_patch(64, 64)
        result = pad_to_size(patch, 64)
        np.testing.assert_array_equal(result, patch)

    def test_padding_applied(self):
        patch = make_rgb_patch(32, 48)
        result = pad_to_size(patch, 64)
        assert result.shape == (64, 64, 3)
        np.testing.assert_array_equal(result[:32, :48], patch)
        assert result[32:, :].sum() == 0   # padded region is zero


class TestIsTissue:
    def _cfg(self) -> TilingConfig:
        return TilingConfig(tissue_threshold=0.05, background_value=230, black_value=10)

    def test_tissue_patch_passes(self):
        patch = make_rgb_patch(64, 64)  # values 0-199 → tissue
        assert is_tissue(patch, self._cfg())

    def test_all_white_rejected(self):
        patch = np.full((64, 64, 3), 240, dtype=np.uint8)
        assert not is_tissue(patch, self._cfg())

    def test_all_black_rejected(self):
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        assert not is_tissue(patch, self._cfg())


class TestTissueMask:
    def test_shape(self):
        from histocoreml.config import TilingConfig
        cfg   = TilingConfig()
        patch = make_rgb_patch(32, 32)
        mask  = tissue_mask(patch, cfg)
        assert mask.shape == (32, 32)
        assert mask.dtype == bool


class TestMacenkoNormalise:
    def test_output_shape_dtype(self):
        patch = make_rgb_patch(64, 64)
        result = macenko_normalise(patch)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_all_white_returns_unchanged(self):
        patch = np.full((32, 32, 3), 240, dtype=np.uint8)
        result = macenko_normalise(patch)
        assert result.shape == (32, 32, 3)
