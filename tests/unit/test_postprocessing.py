"""Tests for histocoreml.postprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from histocoreml.config import ModelConfig, TilingConfig
from histocoreml.io.base_reader import WSIMetadata
from histocoreml.postprocessing.mask_assembler import MaskAssembler
from histocoreml.postprocessing.memmap_canvas import MemmapCanvas
from histocoreml.preprocessing.grid_generator import generate_patch_grid
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
        canvas.counts[:] = 2  # 0.5 average → at 0.5 threshold (included)
        result = canvas.finalise(threshold=0.5)
        assert result.sum() == 16  # 0.5 >= 0.5, so all pixels are 1
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

        metadata = MagicMock()
        # The assembler resolves its level through level_for_mpp so that slides
        # without MPP metadata land on the same level the tiler used.
        metadata.level_for_mpp.return_value = (0, 0.88)
        metadata.best_level_for_mpp.return_value = (0, 0.88)
        metadata.level_downsamples = (1.0,)
        metadata.level_dimensions = ((canvas_w, canvas_h),)

        model_cfg = ModelConfig(model_path="dummy.pt", patch_size=64, target_mpp=0.88)
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
        mask = np.ones((64, 64), dtype=np.uint8)
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


class TestMaskAssemblerScaling:
    """The canvas is at the inference level; model output is at target_mpp.

    When no pyramid level lands exactly on ``target_mpp`` those resolutions
    differ, and predictions must be rescaled before being written. Getting this
    wrong silently double-counts or drops pixels rather than raising.
    """

    @staticmethod
    def _coverage(slide_mpp: float, target_mpp: float, overlap: int, downsample: int = 1):

        metadata = WSIMetadata(
            path=Path("slide.svs"),
            level_count=1,
            level_dimensions=((1024, 1024),),
            level_downsamples=(1.0,),
            mpp_x=slide_mpp,
            mpp_y=slide_mpp,
            vendor=None,
            properties={},
        )
        model_cfg = ModelConfig(
            model_path=Path("model.pt"), patch_size=256, target_mpp=target_mpp, device="cpu"
        )
        tiling_cfg = TilingConfig(overlap=overlap)

        coords = generate_patch_grid(metadata, 256, target_mpp, tiling_cfg)
        assembler = MaskAssembler(metadata, model_cfg, tiling_cfg, downsample_factor=downsample)
        try:
            predictions = np.ones((len(coords), 256, 256), dtype=np.uint8)
            assembler.add_batch(predictions, coords)
            return np.asarray(assembler._canvas.counts).copy(), coords[0].rescale_factor
        finally:
            assembler.cleanup()

    @pytest.mark.parametrize(
        ("slide_mpp", "expected_rescale"),
        [(0.5, 1.0), (0.25, 0.5), (1.0, 2.0)],
    )
    def test_non_overlapping_patches_tile_exactly(self, slide_mpp, expected_rescale):
        counts, rescale = self._coverage(slide_mpp, target_mpp=0.5, overlap=0)

        assert rescale == expected_rescale
        # overlap=0 means a perfect partition: every pixel covered exactly once.
        assert counts.min() == 1
        assert counts.max() == 1

    def test_downsampled_output_still_tiles_exactly(self):
        counts, _ = self._coverage(0.25, target_mpp=0.5, overlap=0, downsample=2)

        assert counts.min() == 1
        assert counts.max() == 1

    def test_overlap_produces_multiple_coverage(self):
        counts, _ = self._coverage(0.25, target_mpp=0.5, overlap=64)

        # Real overlap must still be averaged over, so counts exceed 1 there.
        assert counts.min() == 1
        assert counts.max() > 1

    def test_averaging_is_correct_across_rescaling(self):

        metadata = WSIMetadata(
            path=Path("slide.svs"),
            level_count=1,
            level_dimensions=((512, 512),),
            level_downsamples=(1.0,),
            mpp_x=0.25,
            mpp_y=0.25,
            vendor=None,
            properties={},
        )
        model_cfg = ModelConfig(
            model_path=Path("model.pt"), patch_size=256, target_mpp=0.5, device="cpu"
        )
        tiling_cfg = TilingConfig(overlap=0)
        coords = generate_patch_grid(metadata, 256, 0.5, tiling_cfg)

        assembler = MaskAssembler(metadata, model_cfg, tiling_cfg)
        try:
            assembler.add_batch(np.ones((len(coords), 256, 256), dtype=np.uint8), coords)
            mask = assembler.finalise()
        finally:
            assembler.cleanup()

        # An all-foreground prediction must produce an all-foreground mask.
        assert mask.shape == (512, 512)
        assert mask.min() == 1


class TestMaskAssemblerBatchValidation:
    """Predictions and coordinates must line up or the mask loses patches."""

    @staticmethod
    def _assembler():
        metadata = WSIMetadata(
            path=Path("slide.svs"),
            level_count=1,
            level_dimensions=((512, 512),),
            level_downsamples=(1.0,),
            mpp_x=0.5,
            mpp_y=0.5,
            vendor=None,
            properties={},
        )
        model_cfg = ModelConfig(
            model_path=Path("model.pt"), patch_size=256, target_mpp=0.5, device="cpu"
        )
        tiling_cfg = TilingConfig(overlap=0)
        coords = generate_patch_grid(metadata, 256, 0.5, tiling_cfg)
        return MaskAssembler(metadata, model_cfg, tiling_cfg), coords

    def test_too_few_predictions_raises(self):
        assembler, coords = self._assembler()
        try:
            masks = np.ones((len(coords) - 1, 256, 256), dtype=np.uint8)
            with pytest.raises(ValueError, match="Batch mismatch"):
                assembler.add_batch(masks, coords)
        finally:
            assembler.cleanup()

    def test_too_many_predictions_raises(self):
        assembler, coords = self._assembler()
        try:
            masks = np.ones((len(coords) + 1, 256, 256), dtype=np.uint8)
            with pytest.raises(ValueError, match="Batch mismatch"):
                assembler.add_batch(masks, coords)
        finally:
            assembler.cleanup()

    def test_proba_batch_mismatch_raises(self):
        assembler, coords = self._assembler()
        try:
            probas = np.ones((len(coords) - 1, 256, 256), dtype=np.float32)
            with pytest.raises(ValueError, match="Batch mismatch"):
                assembler.add_proba_batch(probas, coords)
        finally:
            assembler.cleanup()

    def test_matching_batch_is_accepted(self):
        assembler, coords = self._assembler()
        try:
            assembler.add_batch(np.ones((len(coords), 256, 256), dtype=np.uint8), coords)
            assert np.asarray(assembler._canvas.counts).max() == 1
        finally:
            assembler.cleanup()
