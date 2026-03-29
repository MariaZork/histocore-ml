"""Tests for histocoreml.output.rle_codec."""

from __future__ import annotations

import numpy as np
import pytest

from histocoreml.output.rle_codec import (
    CocoRLE, PlainRLE,
    coco_rle_decode, coco_rle_encode,
    encode_patches_to_plain, merge_plain_rles,
    plain_rle_decode, plain_rle_encode,
    plain_rle_from_dict, plain_rle_to_dict,
    coco_rle_from_dict,
    _validate_mask,
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
