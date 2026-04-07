"""Spatial graph construction and statistics from nucleus centroids."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def build_spatial_graph(centroids: List[Tuple[float, float]]) -> Dict:
    """Build a Delaunay triangulation graph from nucleus centroids.

    Args:
        centroids: List of (x, y) centroid coordinates in pixels.

    Returns:
        Dict with keys: edges (list of (i,j) pairs), distances (per edge).
    """
    if len(centroids) < 3:
        return {"edges": [], "distances": []}

    try:
        from scipy.spatial import Delaunay  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    pts = np.array(centroids)

    # Check if all points are collinear (Delaunay will fail)
    if len(pts) >= 2:
        # Check if all points lie on the same line
        x_diff = pts[:, 0] - pts[0, 0]
        y_diff = pts[:, 1] - pts[0, 1]
        # If all x differences are zero (vertical line) or all slopes are the same
        if np.all(x_diff == 0) or np.allclose(y_diff / (x_diff + 1e-10), y_diff[1] / (x_diff[1] + 1e-10), rtol=1e-5):
            # Return edges connecting consecutive points along the line
            sorted_indices = np.argsort(pts[:, 0] + pts[:, 1])
            edges = [(int(sorted_indices[i]), int(sorted_indices[i+1])) for i in range(len(sorted_indices)-1)]
            distances = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in edges]
            return {"edges": edges, "distances": distances}

    try:
        tri = Delaunay(pts)
    except Exception:
        # Fallback for any other Qhull errors - connect nearest neighbors
        edges = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                edges.append((i, j))
        distances = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in edges]
        return {"edges": edges, "distances": distances}

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                edges.add((min(a, b), max(a, b)))

    edges = list(edges)
    distances = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in edges]

    return {"edges": edges, "distances": distances}


def compute_graph_features(graph: Dict, mpp: float = 1.0) -> Dict:
    """Summarise spatial graph properties as scalar features.

    Args:
        graph: Output of :func:`build_spatial_graph`.
        mpp:   Microns-per-pixel for converting distances to µm.

    Returns:
        Dict with keys: mean_dist_um, std_dist_um, median_dist_um,
        num_edges, graph_density.
    """
    distances = np.array(graph["distances"]) * mpp  # convert to µm

    if len(distances) == 0:
        return {
            "mean_dist_um": float("nan"),
            "std_dist_um": float("nan"),
            "median_dist_um": float("nan"),
            "num_edges": 0,
            "graph_density": 0.0,
        }

    n = max(e for pair in graph["edges"] for e in pair) + 1 if graph["edges"] else 0
    max_edges = n * (n - 1) / 2 if n > 1 else 1

    return {
        "mean_dist_um":   float(distances.mean()),
        "std_dist_um":    float(distances.std()),
        "median_dist_um": float(np.median(distances)),
        "num_edges":      len(distances),
        "graph_density":  len(distances) / max_edges,
    }
