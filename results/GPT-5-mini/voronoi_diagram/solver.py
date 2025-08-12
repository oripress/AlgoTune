import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the 2D Voronoi diagram for the given points using SciPy and
        return a dictionary with the expected keys.
        """
        points = problem.get("points", [])
        pts = np.asarray(points, dtype=float)

        # Empty input
        if pts.size == 0:
            return {
                "vertices": [],
                "regions": [],
                "point_region": [],
                "ridge_points": [],
                "ridge_vertices": [],
            }

        # Ensure shape (n,2)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] < 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 2 - pts.shape[1]))])
        else:
            pts = pts[:, :2]

        n_points = int(pts.shape[0])

        # Single point -> one unbounded region
        if n_points == 1:
            return {
                "vertices": [],
                "regions": [[-1]],
                "point_region": [0],
                "ridge_points": [],
                "ridge_vertices": [],
            }

        # Compute Voronoi; jitter deterministically on failure
        try:
            vor = ScipyVoronoi(pts)
        except Exception:
            rng = np.random.default_rng(0)
            scale = 1e-8 * max(1.0, float(np.mean(np.abs(pts))))
            jitter = rng.normal(scale=scale, size=pts.shape)
            vor = ScipyVoronoi(pts + jitter)

        vertices: List[List[float]] = vor.vertices.tolist() if getattr(vor, "vertices", None) is not None else []
        regions_all: List[List[int]] = [list(r) for r in getattr(vor, "regions", [])]
        point_region_raw: List[int] = list(map(int, getattr(vor, "point_region", list(range(n_points)))))
        ridge_points: List[List[int]] = [list(map(int, rp)) for rp in getattr(vor, "ridge_points", [])]
        ridge_vertices: List[List[int]] = [list(map(int, rv)) for rv in getattr(vor, "ridge_vertices", [])]

        # Build regions so regions[i] corresponds to input point i
        regions: List[List[int]] = []
        for idx in point_region_raw:
            if 0 <= idx < len(regions_all):
                regions.append([int(v) for v in regions_all[idx]])
            else:
                regions.append([-1])

        point_region: List[int] = list(range(n_points))

        return {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }