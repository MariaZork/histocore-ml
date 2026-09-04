"""BiomarkerExtractor — orchestrates multi-task biomarker extraction from WSIs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from histocoreml.biomarkers.nuclei import detect_nuclei, measure_nuclei_morphology
from histocoreml.biomarkers.spatial import build_spatial_graph, compute_graph_features
from histocoreml.biomarkers.stain import compute_ki67_index
from histocoreml.config import BiomarkerConfig
from histocoreml.io.factory import get_reader

logger = logging.getLogger(__name__)


@dataclass
class BiomarkerReport:
    """Aggregated biomarker report for a single WSI."""

    wsi_path: Path
    features: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def save(self, path: Path) -> None:
        """Write report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "wsi_path": str(self.wsi_path),
            "features": self.features,
            "errors": self.errors,
            "elapsed_seconds": self.elapsed_seconds,
        }
        with path.open("w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Biomarker report saved → %s", path)

    def to_dict(self) -> dict:
        return {
            "wsi_path": str(self.wsi_path),
            **self.features,
        }


class BiomarkerExtractor:
    """Multi-task biomarker extraction from WSI + segmentation mask.

    Supported tasks (set via BiomarkerConfig.tasks):
    - ``cell_density``      — nuclei per mm²
    - ``nuclei_morphology`` — per-nucleus shape features
    - ``spatial_graph``     — Delaunay graph statistics
    - ``tumor_stroma_ratio``— fraction of tissue in each class
    - ``ki67_index``        — DAB-based proliferation index

    Usage::

        cfg  = BiomarkerConfig(tasks=["cell_density", "spatial_graph"])
        extractor = BiomarkerExtractor(cfg)
        report = extractor.run(Path("slide.svs"), mask=binary_mask)
        report.save(Path("biomarkers/slide.json"))
    """

    def __init__(self, cfg: BiomarkerConfig) -> None:
        self._cfg = cfg

    def run(
        self,
        wsi_path: Path,
        mask: np.ndarray | None = None,
        patch: np.ndarray | None = None,
    ) -> BiomarkerReport:
        """Extract biomarkers from a WSI.

        Args:
            wsi_path: Path to the WSI file.
            mask:     Binary segmentation mask (H, W) uint8. Required for
                      tumor_stroma_ratio. If None, a thumbnail region is used.
            patch:    Optional representative patch (H, W, 3) uint8. If None,
                      a thumbnail-level patch is extracted from the slide.

        Returns:
            :class:`BiomarkerReport` with all requested features.
        """
        t0 = time.perf_counter()
        features: dict[str, Any] = {}
        errors: list[str] = []

        # Get a representative patch for cellular analysis
        if patch is None:
            try:
                with get_reader(wsi_path) as reader:
                    patch = reader.get_thumbnail(max_size=(2048, 2048))
                    meta = reader.get_metadata()
                    mpp = meta.mpp or 1.0
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                errors.append(f"Slide read failed: {exc}")
                return BiomarkerReport(wsi_path=wsi_path, errors=errors)
        else:
            mpp = 1.0  # caller should provide correct mpp separately

        for task in self._cfg.tasks:
            try:
                result = self._run_task(task, patch, mask, mpp)
                features.update(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Task '%s' failed: %s", task, exc, exc_info=True)
                errors.append(f"{task}: {exc}")

        elapsed = time.perf_counter() - t0
        logger.info("Biomarker extraction done for %s in %.1fs", wsi_path.name, elapsed)

        return BiomarkerReport(
            wsi_path=wsi_path,
            features=features,
            errors=errors,
            elapsed_seconds=elapsed,
        )

    def _run_task(
        self,
        task: str,
        patch: np.ndarray,
        mask: np.ndarray | None,
        mpp: float,
    ) -> dict[str, Any]:
        if task == "cell_density":
            _, nuclei = detect_nuclei(patch, self._cfg.min_cell_area_px, self._cfg.max_cell_area_px)
            area_mm2 = (patch.shape[0] * patch.shape[1] * mpp**2) / 1e6
            return {"cell_density_per_mm2": len(nuclei) / max(area_mm2, 1e-9)}

        if task == "nuclei_morphology":
            labelled, _ = detect_nuclei(
                patch,
                self._cfg.min_cell_area_px,
                self._cfg.max_cell_area_px,
            )
            morphs = measure_nuclei_morphology(patch, labelled)
            if morphs:
                return {
                    "mean_nucleus_area_px": float(np.mean([m["area"] for m in morphs])),
                    "mean_eccentricity": float(np.mean([m["eccentricity"] for m in morphs])),
                    "mean_solidity": float(np.mean([m["solidity"] for m in morphs])),
                    "mean_circularity": float(np.mean([m["circularity"] for m in morphs])),
                    "mean_hematoxylin": float(np.mean([m["mean_hematoxylin"] for m in morphs])),
                }
            return {}

        if task == "spatial_graph":
            _, nuclei = detect_nuclei(patch, self._cfg.min_cell_area_px, self._cfg.max_cell_area_px)
            centroids = [n["centroid"] for n in nuclei]
            graph = build_spatial_graph(centroids)
            feats = compute_graph_features(graph, mpp=mpp)
            return {f"spatial_{k}": v for k, v in feats.items()}

        if task == "tumor_stroma_ratio":
            if mask is None:
                return {"tumor_stroma_ratio": float("nan")}
            tumor_px = int(mask.sum())
            total_px = mask.size
            return {
                "tumor_fraction": tumor_px / total_px,
                "stroma_fraction": (total_px - tumor_px) / total_px,
                "tumor_stroma_ratio": tumor_px / max(total_px - tumor_px, 1),
            }

        if task == "ki67_index":
            labelled, _ = detect_nuclei(
                patch,
                self._cfg.min_cell_area_px,
                self._cfg.max_cell_area_px,
            )
            nuclei_mask = (labelled > 0).astype(np.uint8)
            ki67 = compute_ki67_index(patch, nuclei_mask)
            return {"ki67_index": ki67}

        raise ValueError(f"Unknown biomarker task: '{task}'")
