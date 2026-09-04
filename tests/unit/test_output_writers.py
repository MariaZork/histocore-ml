"""Tests for histocoreml.output writers and factory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from histocoreml.config import OutputConfig
from histocoreml.output.factory import get_writer
from histocoreml.output.rle_writer import RLEMaskReader, RLEMaskWriter
from histocoreml.output.writers import NumpyMaskWriter
from tests.conftest import make_binary_mask


def _mock_metadata(mpp: float = 0.88):
    m = MagicMock()
    m.mpp = mpp
    m.path = Path("slide.svs")
    return m


class TestNumpyMaskWriter:
    def test_write_creates_npy(self, tmp_path):
        cfg = OutputConfig(output_dir=tmp_path, output_format="npy")
        writer = NumpyMaskWriter(cfg)
        mask = make_binary_mask(64, 64)
        result = writer.write(mask, _mock_metadata(), stem="test_slide")
        assert result.path.exists()
        assert result.path.suffix == ".npy"
        loaded = np.load(str(result.path))
        np.testing.assert_array_equal(loaded, mask)

    def test_write_result_format(self, tmp_path):
        cfg = OutputConfig(output_dir=tmp_path, output_format="npy")
        writer = NumpyMaskWriter(cfg)
        mask = make_binary_mask(32, 32)
        result = writer.write(mask, _mock_metadata(), stem="slide_001")
        assert result.format == "npy"
        assert result.shape == (32, 32)


class TestRLEMaskWriter:
    def test_plain_roundtrip(self, tmp_path):
        cfg = OutputConfig(output_dir=tmp_path, output_format="rle", rle_subformat="plain")
        writer = RLEMaskWriter(cfg)
        mask = make_binary_mask(64, 64)
        result = writer.write(mask, _mock_metadata(), stem="slide_rle")
        assert result.path.exists()
        reader = RLEMaskReader(result.path)
        decoded = reader.decode()
        np.testing.assert_array_equal(decoded, mask)

    def test_coco_roundtrip(self, tmp_path):
        cfg = OutputConfig(output_dir=tmp_path, output_format="rle", rle_subformat="coco")
        writer = RLEMaskWriter(cfg)
        mask = make_binary_mask(64, 64)
        result = writer.write(mask, _mock_metadata(), stem="slide_coco")
        reader = RLEMaskReader(result.path)
        decoded = reader.decode()
        np.testing.assert_array_equal(decoded, mask)

    def test_mpp_stored_in_json(self, tmp_path):
        cfg = OutputConfig(output_dir=tmp_path, output_format="rle", rle_subformat="plain")
        writer = RLEMaskWriter(cfg)
        mask = make_binary_mask(32, 32)
        result = writer.write(mask, _mock_metadata(mpp=0.50), stem="slide_mpp")
        with result.path.open() as fh:
            data = json.load(fh)
        assert abs(data["mpp"] - 0.50) < 1e-6

    def test_invalid_subformat_raises(self, tmp_path):
        cfg = OutputConfig(output_dir=tmp_path, output_format="rle", rle_subformat="bad")
        writer = RLEMaskWriter(cfg)
        with pytest.raises(ValueError, match="rle_subformat"):
            writer.write(make_binary_mask(16, 16), _mock_metadata(), stem="x")


class TestOutputFactory:
    @pytest.mark.parametrize(
        "fmt,expected_cls",
        [
            ("tiff", "TiffMaskWriter"),
            ("npy", "NumpyMaskWriter"),
            ("rle", "RLEMaskWriter"),
        ],
    )
    def test_factory_returns_correct_writer(self, tmp_path, fmt, expected_cls):
        cfg = OutputConfig(output_dir=tmp_path, output_format=fmt)
        writer = get_writer(cfg)
        assert type(writer).__name__ == expected_cls

    def test_unknown_format_raises(self, tmp_path):
        cfg = OutputConfig(output_dir=tmp_path, output_format="h5")
        with pytest.raises(ValueError, match="Unsupported output format"):
            get_writer(cfg)
