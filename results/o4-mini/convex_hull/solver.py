# pylint: disable=no-name-in-module
from typing import Any
import numpy as np
from scipy.spatial import ConvexHull  # type: ignore

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Convex Hull problem using scipy.spatial.ConvexHull (Quickhull in C).
        """
        pts_list = problem.get("points", [])
        points = np.asarray(pts_list, dtype=float)
        n = points.shape[0]
        # Trivial cases
        if n == 0:
            return {"hull_vertices": [], "hull_points": []}
        if n <= 2:
            verts = list(range(n))
            return {"hull_vertices": verts, "hull_points": points.tolist()}
        # Compute convex hull (C implementation)
        hull = ConvexHull(points)
        vertices = hull.vertices.tolist()
        hull_pts = points[vertices].tolist()
        return {"hull_vertices": vertices, "hull_points": hull_pts}