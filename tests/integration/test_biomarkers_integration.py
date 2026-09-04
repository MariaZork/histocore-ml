"""Integration tests for biomarker extraction.

These tests verify the end-to-end biomarker extraction workflow including:
- Nuclei detection
- Morphology measurement
- Spatial graph analysis
- Stain separation
- Report generation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from histocoreml.biomarkers import BiomarkerExtractor
from histocoreml.biomarkers.nuclei import detect_nuclei, measure_nuclei_morphology
from histocoreml.biomarkers.spatial import build_spatial_graph, compute_graph_features
from histocoreml.biomarkers.stain import (
    compute_ki67_index,
    separate_hdab_channels,
    separate_he_channels,
)
from histocoreml.config import BiomarkerConfig


@pytest.fixture
def synthetic_wsi_patch() -> np.ndarray:
    """Create a synthetic WSI patch with nuclei-like structures."""
    np.random.seed(42)
    patch = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)

    # Add circular nuclei-like structures
    for i in range(15):
        cx, cy = np.random.randint(30, 226, 2)
        radius = np.random.randint(8, 20)
        color = [60, 60, 120]  # Dark blue-ish

        y, x = np.ogrid[:256, :256]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        patch[mask] = color

    return patch


@pytest.fixture
def synthetic_mask() -> np.ndarray:
    """Create a synthetic binary mask."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Create a tumor region
    mask[50:150, 50:150] = 1
    # Create another tumor region
    mask[180:230, 180:230] = 1
    return mask


@pytest.fixture
def small_wsi_file(tmp_path: Path) -> Path:
    """Create a small synthetic WSI file."""
    path = tmp_path / "biomarker_test.tiff"

    np.random.seed(42)
    img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)

    with tifffile.TiffWriter(path) as writer:
        writer.write(img)

    return path


class TestBiomarkerExtractorIntegration:
    """Integration tests for the full biomarker extraction pipeline."""

    def test_extractor_with_cell_density_task(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Extractor should compute cell density."""
        cfg = BiomarkerConfig(tasks=["cell_density"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)

        assert "cell_density_per_mm2" in report.features
        assert report.features["cell_density_per_mm2"] > 0

    def test_extractor_with_nuclei_morphology_task(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Extractor should compute morphology features."""
        cfg = BiomarkerConfig(tasks=["nuclei_morphology"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)

        assert "mean_nucleus_area_px" in report.features
        assert "mean_eccentricity" in report.features
        assert "mean_solidity" in report.features
        assert "mean_circularity" in report.features
        assert "mean_hematoxylin" in report.features

    def test_extractor_with_spatial_graph_task(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Extractor should compute spatial graph features."""
        cfg = BiomarkerConfig(tasks=["spatial_graph"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)

        # Should have spatial graph features
        spatial_features = [k for k in report.features.keys() if k.startswith("spatial_")]
        assert len(spatial_features) > 0

    def test_extractor_with_tumor_stroma_ratio(
        self, synthetic_wsi_patch: np.ndarray, synthetic_mask: np.ndarray
    ) -> None:
        """Extractor should compute tumor/stroma ratio from mask."""
        cfg = BiomarkerConfig(tasks=["tumor_stroma_ratio"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(
            Path("/fake/path.tiff"), patch=synthetic_wsi_patch, mask=synthetic_mask
        )

        assert "tumor_fraction" in report.features
        assert "stroma_fraction" in report.features
        assert "tumor_stroma_ratio" in report.features

        # Check values are in valid range
        assert 0 <= report.features["tumor_fraction"] <= 1
        assert 0 <= report.features["stroma_fraction"] <= 1

    def test_extractor_with_multiple_tasks(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Extractor should handle multiple tasks in one run."""
        cfg = BiomarkerConfig(tasks=["cell_density", "nuclei_morphology"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)

        assert "cell_density_per_mm2" in report.features
        assert "mean_nucleus_area_px" in report.features

    def test_report_save_and_load(self, synthetic_wsi_patch: np.ndarray, tmp_path: Path) -> None:
        """Report should save to JSON and reload correctly."""
        cfg = BiomarkerConfig(tasks=["cell_density"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)
        output_path = tmp_path / "report.json"
        report.save(output_path)

        assert output_path.exists()

        # Load and verify
        with open(output_path) as f:
            data = json.load(f)

        assert "wsi_path" in data
        assert "features" in data
        assert "cell_density_per_mm2" in data["features"]

    def test_report_success_flag(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Report should have success flag based on errors."""
        cfg = BiomarkerConfig(tasks=["cell_density"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)

        assert report.success == (len(report.errors) == 0)

    def test_extractor_with_empty_patch(self) -> None:
        """Extractor should handle empty patches gracefully."""
        cfg = BiomarkerConfig(tasks=["cell_density"])
        extractor = BiomarkerExtractor(cfg)

        empty_patch = np.full((256, 256, 3), 255, dtype=np.uint8)  # All white

        report = extractor.run(Path("/fake/path.tiff"), patch=empty_patch)

        # Should complete without crash, possibly with zero density
        assert report.success

    def test_extractor_timing(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Report should include elapsed time."""
        cfg = BiomarkerConfig(tasks=["cell_density"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)

        assert report.elapsed_seconds >= 0


class TestNucleiDetectionIntegration:
    """Integration tests for nuclei detection."""

    def test_detect_nuclei_on_synthetic(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should detect nuclei in synthetic patch."""
        labelled, nuclei = detect_nuclei(synthetic_wsi_patch)

        assert isinstance(labelled, np.ndarray)
        assert isinstance(nuclei, list)
        assert len(nuclei) > 0

    def test_nuclei_have_centroids(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Detected nuclei should have centroids."""
        _, nuclei = detect_nuclei(synthetic_wsi_patch)

        for nucleus in nuclei:
            assert "centroid" in nucleus
            assert len(nucleus["centroid"]) == 2

    def test_measure_nuclei_morphology(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should compute morphology for detected nuclei."""
        labelled, _ = detect_nuclei(synthetic_wsi_patch)
        morphs = measure_nuclei_morphology(synthetic_wsi_patch, labelled)

        assert len(morphs) > 0
        for morph in morphs:
            assert "area" in morph
            assert "eccentricity" in morph
            assert "solidity" in morph
            assert "circularity" in morph
            assert "mean_hematoxylin" in morph


class TestSpatialGraphIntegration:
    """Integration tests for spatial graph analysis."""

    def test_build_spatial_graph(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should build graph from nuclei centroids."""
        _, nuclei = detect_nuclei(synthetic_wsi_patch)
        centroids = [n["centroid"] for n in nuclei]

        graph = build_spatial_graph(centroids)

        assert graph is not None
        assert graph["num_nodes"] == len(centroids)
        # Every edge references a valid centroid index.
        assert all(0 <= i < len(centroids) and 0 <= j < len(centroids) for i, j in graph["edges"])
        assert len(graph["distances"]) == len(graph["edges"])

    def test_compute_graph_features(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should compute graph features."""
        _, nuclei = detect_nuclei(synthetic_wsi_patch)
        centroids = [n["centroid"] for n in nuclei]

        if len(centroids) >= 3:
            graph = build_spatial_graph(centroids)
            feats = compute_graph_features(graph, mpp=0.5)

            assert feats["num_nodes"] == len(centroids)
            assert feats["num_edges"] == len(graph["edges"])
            assert "avg_degree" in feats
            assert 0.0 <= feats["graph_density"] <= 1.0

    def test_spatial_graph_few_nuclei(self) -> None:
        """Too few nuclei yields an empty graph, not None."""
        graph = build_spatial_graph([(50, 50)])

        assert graph["edges"] == []
        assert graph["num_nodes"] == 1

        feats = compute_graph_features(graph)
        assert feats["num_edges"] == 0
        assert feats["graph_density"] == 0.0

    def test_graph_density_accounts_for_isolated_nodes(self) -> None:
        """Density must divide by the real node count, not the highest edge index."""
        # Two tight clusters: every node participates, so density stays <= 1.
        centroids = [(0, 0), (10, 0), (5, 8), (200, 200), (210, 200), (205, 208)]
        feats = compute_graph_features(build_spatial_graph(centroids))

        assert feats["num_nodes"] == len(centroids)
        assert 0.0 <= feats["graph_density"] <= 1.0

    def test_collinear_centroids_form_a_chain(self) -> None:
        """Points on a diagonal line must not fall back to a complete graph."""
        diagonal = build_spatial_graph([(0, 0), (1, 1), (2, 2), (3, 3)])
        horizontal = build_spatial_graph([(0, 0), (1, 0), (2, 0), (3, 0)])

        assert len(diagonal["edges"]) == len(horizontal["edges"]) == 3


class TestStainSeparationIntegration:
    """Integration tests for stain separation."""

    def test_separate_he_stains(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should separate H&E stains."""
        hematoxylin, eosin = separate_he_channels(synthetic_wsi_patch)

        assert hematoxylin.shape == synthetic_wsi_patch.shape[:2]
        assert eosin.shape == synthetic_wsi_patch.shape[:2]
        assert hematoxylin.dtype == np.float32
        assert eosin.dtype == np.float32

    def test_separate_hdab_stains(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should separate HDAB stains."""
        hematoxylin, dab = separate_hdab_channels(synthetic_wsi_patch)

        assert hematoxylin.shape == synthetic_wsi_patch.shape[:2]
        assert dab.shape == synthetic_wsi_patch.shape[:2]
        assert hematoxylin.dtype == np.float32
        assert dab.dtype == np.float32

    def test_compute_ki67_index(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should compute Ki67 index."""
        _, nuclei = detect_nuclei(synthetic_wsi_patch)
        nuclei_mask = np.zeros(synthetic_wsi_patch.shape[:2], dtype=np.uint8)
        for n in nuclei:
            cy, cx = int(n["centroid"][0]), int(n["centroid"][1])
            if 0 <= cy < nuclei_mask.shape[0] and 0 <= cx < nuclei_mask.shape[1]:
                nuclei_mask[cy, cx] = 1

        ki67 = compute_ki67_index(synthetic_wsi_patch, nuclei_mask)

        # Result should be a float
        assert isinstance(ki67, (float, np.floating))


class TestBiomarkerEdgeCases:
    """Tests for edge cases in biomarker extraction."""

    def test_extractor_unknown_task(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Should report error for unknown tasks."""
        cfg = BiomarkerConfig(tasks=["unknown_task"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)

        # Should have errors for unknown task
        assert len(report.errors) > 0 or not report.success

    def test_morphology_empty_mask(self) -> None:
        """Should handle empty mask."""
        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        patch = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        morphs = measure_nuclei_morphology(patch, empty_mask)

        assert morphs == []

    def test_ki67_empty_nuclei_mask(self) -> None:
        """Should handle empty nuclei mask for Ki67."""
        patch = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        empty_mask = np.zeros((256, 256), dtype=np.uint8)

        ki67 = compute_ki67_index(patch, empty_mask)

        assert ki67 == 0.0 or np.isnan(ki67)

    def test_report_to_dict(self, synthetic_wsi_patch: np.ndarray) -> None:
        """Report should convert to dict."""
        cfg = BiomarkerConfig(tasks=["cell_density"])
        extractor = BiomarkerExtractor(cfg)

        report = extractor.run(Path("/fake/path.tiff"), patch=synthetic_wsi_patch)
        d = report.to_dict()

        assert "wsi_path" in d
        assert "cell_density_per_mm2" in d
