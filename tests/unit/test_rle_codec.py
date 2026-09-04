"""Tests for histocoreml.output.rle_codec."""

from __future__ import annotations

import numpy as np
import pytest

from histocoreml.output.rle_codec import (
    _validate_mask,
    coco_rle_decode,
    coco_rle_encode,
    coco_rle_from_dict,
    encode_patches_to_plain,
    plain_rle_decode,
    plain_rle_encode,
    plain_rle_from_dict,
    plain_rle_to_dict,
)
from tests.conftest import make_binary_mask


class TestPlainRLE:
    def test_roundtrip(self):
        mask = make_binary_mask(64, 64)
        rle = plain_rle_encode(mask)
        decoded = plain_rle_decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_all_zeros(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        rle = plain_rle_encode(mask)
        assert len(rle.runs) == 1
        assert rle.runs[0] == (0, 32 * 32)
        np.testing.assert_array_equal(rle.decode(), mask)

    def test_all_ones(self):
        mask = np.ones((32, 32), dtype=np.uint8)
        rle = plain_rle_encode(mask)
        assert len(rle.runs) == 1
        assert rle.runs[0] == (1, 32 * 32)

    def test_serialisation_roundtrip(self):
        mask = make_binary_mask(32, 32)
        rle = plain_rle_encode(mask)
        d = plain_rle_to_dict(rle)
        rle2 = plain_rle_from_dict(d)
        np.testing.assert_array_equal(rle2.decode(), mask)

    def test_compression_ratio_positive(self):
        mask = make_binary_mask(64, 64, ratio=0.3)
        rle = plain_rle_encode(mask)
        assert rle.compression_ratio() > 0

    def test_total_foreground(self):
        mask = make_binary_mask(32, 32, ratio=0.5)
        rle = plain_rle_encode(mask)
        assert rle.total_foreground() == int(mask.sum())


class TestCocoRLE:
    def test_roundtrip(self):
        mask = make_binary_mask(64, 64)
        rle = coco_rle_encode(mask)
        decoded = coco_rle_decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_starts_with_background(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        rle = coco_rle_encode(mask)
        assert rle.counts[0] == 0

    def test_coco_dict_roundtrip(self):
        mask = make_binary_mask(32, 32)
        rle = coco_rle_encode(mask)
        d = rle.to_coco_dict()
        rle2 = coco_rle_from_dict(d)
        np.testing.assert_array_equal(coco_rle_decode(rle2), mask)


class TestValidateMask:
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            _validate_mask(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_rejects_values_above_one(self):
        with pytest.raises(ValueError, match="values must be 0 or 1"):
            _validate_mask(np.array([[0, 1, 2]], dtype=np.uint8))


class TestStreamingHelpers:
    def test_encode_patches_to_plain(self):
        masks = make_binary_mask(64, 64).reshape(1, 64, 64)
        masks = np.stack([masks[0]] * 3)
        rles = encode_patches_to_plain(masks)
        assert len(rles) == 3
        for rle in rles:
            np.testing.assert_array_equal(rle.decode(), masks[0])


class TestEncodePatchesToPlain:
    """Batch encoding — previously shipped with no coverage at all."""

    def test_encodes_one_rle_per_patch(self):
        from histocoreml.output.rle_codec import encode_patches_to_plain

        masks = np.zeros((3, 8, 8), dtype=np.uint8)
        masks[1, 2:5, 2:5] = 1

        rles = encode_patches_to_plain(masks)

        assert len(rles) == 3
        assert all(r.shape == (8, 8) for r in rles)

    def test_each_patch_round_trips(self):
        from histocoreml.output.rle_codec import encode_patches_to_plain

        rng = np.random.default_rng(0)
        masks = (rng.random((4, 16, 16)) > 0.5).astype(np.uint8)

        for original, rle in zip(masks, encode_patches_to_plain(masks), strict=True):
            np.testing.assert_array_equal(rle.decode(), original)

    def test_empty_batch(self):
        from histocoreml.output.rle_codec import encode_patches_to_plain

        assert encode_patches_to_plain(np.zeros((0, 8, 8), dtype=np.uint8)) == []


class TestMergePlainRLEs:
    """Reassembling patch RLEs onto a slide canvas."""

    @staticmethod
    def _coord(x: int, y: int, size: int = 8):
        from histocoreml.preprocessing.patch_coord import PatchCoord

        return PatchCoord(x=x, y=y, level=0, patch_size=size, col_idx=0, row_idx=0)

    def _pairs(self, masks, positions, size=8):
        from histocoreml.output.rle_codec import plain_rle_encode

        return [
            (self._coord(x, y, size), plain_rle_encode(m))
            for m, (x, y) in zip(masks, positions, strict=True)
        ]

    def test_patches_land_at_their_coordinates(self):
        from histocoreml.output.rle_codec import merge_plain_rles

        left = np.zeros((8, 8), dtype=np.uint8)
        left[:, :] = 1
        right = np.zeros((8, 8), dtype=np.uint8)

        canvas = merge_plain_rles(self._pairs([left, right], [(0, 0), (8, 0)]), 8, 16)

        assert canvas.shape == (8, 16)
        assert canvas[:, :8].all()
        assert not canvas[:, 8:].any()

    def test_overlapping_patches_are_unioned(self):
        """Unlike MaskAssembler, this ORs overlaps rather than averaging them."""
        from histocoreml.output.rle_codec import merge_plain_rles

        a = np.zeros((8, 8), dtype=np.uint8)
        a[0:4, :] = 1
        b = np.zeros((8, 8), dtype=np.uint8)
        b[4:8, :] = 1

        canvas = merge_plain_rles(self._pairs([a, b], [(0, 0), (0, 0)]), 8, 8)

        assert canvas.all()  # union of the two halves covers everything

    def test_patches_are_clipped_to_the_canvas(self):
        from histocoreml.output.rle_codec import merge_plain_rles

        full = np.ones((8, 8), dtype=np.uint8)

        canvas = merge_plain_rles(self._pairs([full], [(6, 6)]), 8, 8)

        assert canvas[6:, 6:].all()
        assert canvas.sum() == 4  # only the 2x2 corner fits

    def test_patch_entirely_outside_is_dropped(self):
        from histocoreml.output.rle_codec import merge_plain_rles

        full = np.ones((8, 8), dtype=np.uint8)

        canvas = merge_plain_rles(self._pairs([full], [(100, 100)]), 8, 8)

        assert not canvas.any()

    def test_downsample_factor_scales_coordinates(self):
        from histocoreml.output.rle_codec import merge_plain_rles

        full = np.ones((4, 4), dtype=np.uint8)

        # A level-0 coordinate of 8 lands at 4 on a canvas downsampled by 2.
        canvas = merge_plain_rles(self._pairs([full], [(8, 0)], size=4), 4, 8, downsample_factor=2)

        assert canvas[:, 4:8].all()
        assert not canvas[:, :4].any()

    def test_empty_input_gives_empty_canvas(self):
        from histocoreml.output.rle_codec import merge_plain_rles

        canvas = merge_plain_rles([], 8, 8)

        assert canvas.shape == (8, 8)
        assert not canvas.any()
