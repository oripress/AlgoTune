from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Compute Delaunay triangulation for 2D points using SciPy's Qhull backend.

        Returns:
            dict with:
              - "simplices": numpy array of shape (m, 3) with triangle vertex indices
              - "convex_hull": numpy array of shape (k, 2) with hull edge indices
        """
        pts_in = problem["points"]
        n = len(pts_in)

        # Fast path for exactly 3 non-collinear points
        if n == 3:
            a = np.asarray(pts_in, dtype=float)
            # Check collinearity; fallback to Qhull if degenerate
            x0, y0 = a[0]
            x1, y1 = a[1]
            x2, y2 = a[2]
            area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if area2 != 0.0:
                simplices = np.array([[0, 1, 2]], dtype=int)
                convex_hull = np.array([[0, 1], [1, 2], [0, 2]], dtype=int)
                return {"simplices": simplices, "convex_hull": convex_hull}
            # else degenerate -> fall through to SciPy (will raise as in reference)

        # Prepare contiguous float64 array only if needed
        if isinstance(pts_in, np.ndarray) and pts_in.dtype == np.float64 and pts_in.flags.c_contiguous:
            pts = pts_in
        else:
            pts = np.ascontiguousarray(pts_in, dtype=np.float64)

        tri = SciPyDelaunay(pts)
        return {
            "simplices": tri.simplices,
            "convex_hull": tri.convex_hull,
        }