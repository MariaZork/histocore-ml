"""Tests for histocoreml.postprocessing."""

from __future__ import annotations

import numpy as np

from histocoreml.postprocessing.mask_assembler import MaskAssembler
from histocoreml.postprocessing.memmap_canvas import MemmapCanvas
from histocoreml.preprocessing.patch_coord import PatchCoord


class TestMemmapCanvas:
    def test_create_and_shape(self):
        canvas = MemmapCanvas.create(128, 256)
        assert canvas.shape == (128, 256)
        assert canvas.accumulator.shape == (128, 256)
        assert canvas.counts.shape == (128, 256)
        canvas.cleanup()

    def test_finalise_empty(self):
        canvas = MemmapCanvas.create(32, 32)
        result = canvas.finalise()
        assert result.shape == (32, 32)
        assert result.sum() == 0
        canvas.cleanup()

    def test_finalise_all_ones(self):
        canvas = MemmapCanvas.create(32, 32)
        canvas.accumulator[:] = 1
        canvas.counts[:] = 1
        result = canvas.finalise()
        np.testing.assert_array_equal(result, np.ones((32, 32), dtype=np.uint8))
        canvas.cleanup()

    def test_averaging_overlap(self):
        canvas = MemmapCanvas.create(4, 4)
        canvas.accumulator[:] = 1
        canvas.counts[:] = 2          # 0.5 average → at 0.5 threshold (included)
        result = canvas.finalise(threshold=0.5)
        assert result.sum() == 16     # 0.5 >= 0.5, so all pixels are 1
        canvas.cleanup()

    def test_cleanup_removes_files(self):
        canvas = MemmapCanvas.create(16, 16)
        tmpdir = canvas.tmpdir
        assert tmpdir.exists()
        canvas.cleanup()
        assert not tmpdir.exists()


class TestMaskAssemblerUnit:
    """Unit tests for _write_patch logic using a minimal mock metadata."""

    def _make_assembler(self, canvas_h=128, canvas_w=128):
        """Build a MaskAssembler with a mocked WSIMetadata."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from histocoreml.config import ModelConfig, TilingConfig  # noqa: PLC0415

        metadata = MagicMock()
        metadata.best_level_for_mpp.return_value = (0, 0.88)
        metadata.level_downsamples = (1.0,)
        metadata.level_dimensions = ((canvas_w, canvas_h),)

        model_cfg  = ModelConfig(model_path="dummy.pt", patch_size=64, target_mpp=0.88)
        tiling_cfg = TilingConfig()
        return MaskAssembler(metadata, model_cfg, tiling_cfg, downsample_factor=1)

    def test_single_patch_written(self):
        assembler = self._make_assembler(128, 128)
        mask = np.ones((64, 64), dtype=np.uint8)
        coord = PatchCoord(x=0, y=0, level=0, patch_size=64, col_idx=0, row_idx=0)
        assembler.add_batch(np.array([mask]), [coord])
        result = assembler.finalise()
        assert result[:64, :64].sum() == 64 * 64
        assert result[64:, :].sum() == 0
        assembler.cleanup()

    def test_out_of_bounds_patch_ignored(self):
        assembler = self._make_assembler(32, 32)
        mask  = np.ones((64, 64), dtype=np.uint8)
        coord = PatchCoord(x=200, y=200, level=0, patch_size=64, col_idx=99, row_idx=99)
        assembler.add_batch(np.array([mask]), [coord])
        result = assembler.finalise()
        assert result.sum() == 0
        assembler.cleanup()

    def test_overlapping_patches_averaged(self):
        assembler = self._make_assembler(64, 64)
        patch_a = np.ones((32, 32), dtype=np.uint8)
        patch_b = np.zeros((32, 32), dtype=np.uint8)
        coord = PatchCoord(x=0, y=0, level=0, patch_size=32, col_idx=0, row_idx=0)
        assembler.add_batch(np.array([patch_a]), [coord])
        assembler.add_batch(np.array([patch_b]), [coord])
        # average = 0.5, threshold = 0.5 → binarised to 1
        result = assembler.finalise()
        assert result[:32, :32].sum() == 32 * 32
        assembler.cleanup()
