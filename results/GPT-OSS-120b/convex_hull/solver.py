import logging
from typing import Any, List, Dict
import numpy as np
from scipy.spatial import ConvexHull, QhullError
class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convex hull.
        For small inputs we use a pure‑Python monotonic chain (O(n log n));
        for large inputs we fall back to SciPy's highly‑optimized C implementation.
        Returns hull vertex indices and points in counter‑clockwise order.
        Handles trivial and collinear cases.
        """
        points: List[List[float]] = problem.get("points", [])
        n = len(points)

        # Trivial cases
        if n == 0:
            return {"hull_vertices": [], "hull_points": []}
        if n == 1:
            return {"hull_vertices": [0], "hull_points": [points[0]]}
        if n == 2:
            return {"hull_vertices": [0, 1], "hull_points": [points[0], points[1]]}

        # Use SciPy for large datasets (reduces Python overhead)
        if n > 2000:
            pts_arr = np.asarray(points, dtype=float)
            try:
                hull = ConvexHull(pts_arr)
                hull_vertices = hull.vertices.tolist()
            except QhullError:
                # Collinear case: extreme points only
                sorted_idx = np.lexsort((pts_arr[:, 1], pts_arr[:, 0]))
                hull_vertices = [int(sorted_idx[0]), int(sorted_idx[-1])]
            hull_points = pts_arr[hull_vertices].tolist()
            return {"hull_vertices": hull_vertices, "hull_points": hull_points}

        # ---- Pure‑Python monotonic chain for smaller inputs ----
        # Attach original indices
        indexed = [(pt[0], pt[1], idx) for idx, pt in enumerate(points)]
        # Sort by x then y
        indexed.sort(key=lambda x: (x[0], x[1]))

        def cross(o, a, b):
            """Cross product (OA x OB) z‑component."""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in indexed:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(indexed):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenate, removing duplicate endpoints
        hull = lower[:-1] + upper[:-1]

        hull_vertices = [p[2] for p in hull]
        hull_points = [[p[0], p[1]] for p in hull]

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}