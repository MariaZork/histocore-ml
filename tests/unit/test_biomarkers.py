"""Tests for histocoreml.biomarkers."""

from __future__ import annotations

import numpy as np

from histocoreml.biomarkers.nuclei import detect_nuclei, measure_nuclei_morphology
from histocoreml.biomarkers.spatial import build_spatial_graph, compute_graph_features
from histocoreml.biomarkers.stain import (
    compute_ki67_index,
    separate_hdab_channels,
    separate_he_channels,
)
from tests.conftest import make_rgb_patch


class TestNucleiDetection:
    def test_returns_labelled_mask_and_list(self):
        patch = make_rgb_patch(128, 128)
        labels, nuclei = detect_nuclei(patch, min_area=10, max_area=10000)
        assert labels.shape == (128, 128)
        assert isinstance(nuclei, list)

    def test_all_white_patch_no_nuclei(self):
        patch = np.full((64, 64, 3), 240, dtype=np.uint8)
        _, nuclei = detect_nuclei(patch)
        assert len(nuclei) == 0


class TestNucleiMorphology:
    def test_returns_list_of_dicts(self):
        patch = make_rgb_patch(64, 64)
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[10:20, 10:20] = 1  # fake nucleus
        labels[30:40, 30:40] = 2
        feats = measure_nuclei_morphology(patch, labels)
        assert len(feats) == 2
        for f in feats:
            assert "area" in f
            assert "eccentricity" in f

    def test_empty_mask_returns_empty(self):
        patch = make_rgb_patch(32, 32)
        labels = np.zeros((32, 32), dtype=np.int32)
        feats = measure_nuclei_morphology(patch, labels)
        assert feats == []


class TestSpatialGraph:
    def test_triangle_produces_three_edges(self):
        centroids = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)]
        graph = build_spatial_graph(centroids)
        assert len(graph["edges"]) == 3

    def test_too_few_points(self):
        graph = build_spatial_graph([(0.0, 0.0), (1.0, 1.0)])
        assert graph["edges"] == []

    def test_graph_features_keys(self):
        centroids = [(float(i), float(i)) for i in range(10)]
        graph = build_spatial_graph(centroids)
        feats = compute_graph_features(graph, mpp=0.25)
        for key in ["mean_dist_um", "std_dist_um", "median_dist_um", "num_edges"]:
            assert key in feats


class TestStainSeparation:
    def test_he_channels_shape(self):
        patch = make_rgb_patch(64, 64)
        h, e = separate_he_channels(patch)
        assert h.shape == (64, 64)
        assert e.shape == (64, 64)
        assert h.dtype == np.float32

    def test_hdab_channels_shape(self):
        patch = make_rgb_patch(32, 32)
        h, d = separate_hdab_channels(patch)
        assert h.shape == (32, 32)
        assert d.shape == (32, 32)

    def test_ki67_empty_mask_returns_nan(self):
        patch = make_rgb_patch(32, 32)
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = compute_ki67_index(patch, mask)
        assert np.isnan(result)

    def test_ki67_range(self):
        patch = make_rgb_patch(32, 32)
        mask = np.ones((32, 32), dtype=np.uint8)
        result = compute_ki67_index(patch, mask)
        assert 0.0 <= result <= 1.0


class TestLongevityAnalyzer:
    """LongevityAnalyzer is public API but had no coverage.

    Its fibrosis step called ``morphology.regionprops``, which does not exist —
    regionprops lives in ``skimage.measure`` — so every analyze() raised
    AttributeError.
    """

    @staticmethod
    def _patch() -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.integers(60, 200, (256, 256, 3), dtype=np.uint8)

    def test_detect_fibrosis_runs(self):
        from histocoreml.biomarkers.longevity import LongevityAnalyzer

        result = LongevityAnalyzer()._detect_fibrosis(self._patch())

        assert 0.0 <= result["index"] <= 100.0
        assert result["area_px"] >= 0
        assert isinstance(result["histogram"], list)

    def test_detect_senescence_runs(self):
        from histocoreml.biomarkers.longevity import LongevityAnalyzer

        result = LongevityAnalyzer()._detect_senescence(self._patch())

        assert result["total_cells"] >= 0
        assert result["senescent_cells"] <= result["total_cells"]

    def test_analyze_end_to_end(self, tmp_path):
        import tifffile

        from histocoreml.biomarkers.longevity import LongevityAnalyzer, LongevityMetrics

        path = tmp_path / "slide.tiff"
        tifffile.imwrite(path, self._patch())

        metrics = LongevityAnalyzer().analyze(path)

        assert isinstance(metrics, LongevityMetrics)
        assert 0.0 <= metrics.health_score <= 100.0
        assert metrics.total_tissue_area_px > 0
