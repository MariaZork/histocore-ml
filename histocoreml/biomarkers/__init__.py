"""HistoCoreML biomarkers — quantitative feature extraction from histology.

Biomarkers extracted
--------------------
* **Cell density**       — nuclei per mm² in tissue regions
* **Nuclei morphology**  — area, eccentricity, solidity per nucleus
* **Tumor-stroma ratio** — foreground class pixel fractions
* **Spatial graph**      — Delaunay graph statistics (mean neighbour distance, etc.)
* **Ki-67 index**        — fraction of positively-staining nuclei (DAB channel)
* **Longevity metrics**  — Tissue health score, fibrosis, senescence, epigenetic age

Usage::

    from histocoreml.biomarkers import BiomarkerExtractor
    from histocoreml.config import BiomarkerConfig
    from pathlib import Path

    cfg = BiomarkerConfig(tasks=["cell_density", "nuclei_morphology"], target_mpp=0.25)
    extractor = BiomarkerExtractor(cfg)
    report = extractor.run(Path("slide.svs"), mask=binary_mask)
    report.save(Path("biomarkers/slide.json"))

    # Longevity analysis
    from histocoreml.biomarkers import LongevityAnalyzer
    analyzer = LongevityAnalyzer()
    metrics = analyzer.analyze(Path("slide.svs"))
"""

from histocoreml.biomarkers.extractor import BiomarkerExtractor, BiomarkerReport
from histocoreml.biomarkers.longevity import (
    LongevityAnalyzer,
    LongevityMetrics,
    analyze_tissue_longevity,
)
from histocoreml.biomarkers.nuclei import detect_nuclei, measure_nuclei_morphology
from histocoreml.biomarkers.spatial import build_spatial_graph, compute_graph_features
from histocoreml.biomarkers.stain import compute_ki67_index, separate_he_channels

__all__ = [
    "BiomarkerExtractor",
    "BiomarkerReport",
    "LongevityAnalyzer",
    "LongevityMetrics",
    "analyze_tissue_longevity",
    "detect_nuclei",
    "measure_nuclei_morphology",
    "build_spatial_graph",
    "compute_graph_features",
    "separate_he_channels",
    "compute_ki67_index",
]
