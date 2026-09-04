"""Spatial graph construction and statistics from nucleus centroids."""

from __future__ import annotations

import numpy as np


def build_spatial_graph(centroids: list[tuple[float, float]]) -> dict:
    """Build a Delaunay triangulation graph from nucleus centroids.

    Args:
        centroids: List of (x, y) centroid coordinates in pixels.

    Returns:
        Dict with keys: ``edges`` (list of ``(i, j)`` index pairs),
        ``distances`` (pixel distance per edge) and ``num_nodes``. Fewer than
        three centroids cannot be triangulated, so the graph comes back with no
        edges rather than being ``None`` — callers always get the same shape.
    """
    if len(centroids) < 3:
        return {"edges": [], "distances": [], "num_nodes": len(centroids)}

    try:
        from scipy.spatial import Delaunay  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    pts = np.array(centroids)
    num_nodes = len(pts)

    # Check if all points are collinear (Delaunay will fail).
    if len(pts) >= 2:
        x_diff = pts[:, 0] - pts[0, 0]
        y_diff = pts[:, 1] - pts[0, 1]
        # Slope of every point relative to pts[0]. Index 0 is 0/0 and carries no
        # information, so compare the rest among themselves — comparing against
        # it would only ever detect horizontal lines.
        slopes = y_diff[1:] / (x_diff[1:] + 1e-10)
        if np.all(x_diff == 0) or np.allclose(slopes, slopes[0], rtol=1e-5):
            # Return edges connecting consecutive points along the line
            sorted_indices = np.argsort(pts[:, 0] + pts[:, 1])
            line_edges: list[tuple[int, int]] = [
                (int(sorted_indices[i]), int(sorted_indices[i + 1]))
                for i in range(len(sorted_indices) - 1)
            ]
            distances = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in line_edges]
            return {"edges": line_edges, "distances": distances, "num_nodes": num_nodes}
    try:
        from scipy.spatial import QhullError  # noqa: PLC0415

        tri = Delaunay(pts)
    except QhullError:
        # Fallback for any other Qhull errors - connect nearest neighbors
        fallback_edges: list[tuple[int, int]] = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                fallback_edges.append((i, j))
        distances = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in fallback_edges]
        return {"edges": fallback_edges, "distances": distances, "num_nodes": num_nodes}

    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                edge_set.add((min(a, b), max(a, b)))

    edges = list(edge_set)
    distances = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in edges]

    return {"edges": edges, "distances": distances, "num_nodes": num_nodes}


def compute_graph_features(graph: dict, mpp: float = 1.0) -> dict:
    """Summarise spatial graph properties as scalar features.

    Args:
        graph: Output of :func:`build_spatial_graph`.
        mpp:   Microns-per-pixel for converting distances to µm.

    Returns:
        Dict with keys: mean_dist_um, std_dist_um, median_dist_um,
        num_nodes, num_edges, avg_degree, graph_density.
    """
    distances = np.array(graph["distances"]) * mpp  # convert to µm
    num_nodes = _node_count(graph)

    if len(distances) == 0:
        return {
            "mean_dist_um": float("nan"),
            "std_dist_um": float("nan"),
            "median_dist_um": float("nan"),
            "num_nodes": num_nodes,
            "num_edges": 0,
            "avg_degree": 0.0,
            "graph_density": 0.0,
        }

    max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1

    return {
        "mean_dist_um": float(distances.mean()),
        "std_dist_um": float(distances.std()),
        "median_dist_um": float(np.median(distances)),
        "num_nodes": num_nodes,
        "num_edges": len(distances),
        # Each edge contributes to the degree of both endpoints.
        "avg_degree": (2 * len(distances) / num_nodes) if num_nodes else 0.0,
        "graph_density": len(distances) / max_edges,
    }


def _node_count(graph: dict) -> int:
    """Node count for *graph*, tolerating dicts built before ``num_nodes`` existed.

    Inferring it from the highest edge index undercounts whenever the
    last centroids are isolated, which skews ``graph_density``; it is only a
    fallback for graphs that predate the explicit field.
    """
    if "num_nodes" in graph:
        return int(graph["num_nodes"])
    if not graph["edges"]:
        return 0
    return max(e for pair in graph["edges"] for e in pair) + 1
