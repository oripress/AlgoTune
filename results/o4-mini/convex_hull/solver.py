from typing import Any, Dict
import numpy as np
from scipy.spatial.qhull import ConvexHull

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the Convex Hull problem using scipy.spatial.qhull.ConvexHull.
        """
        pts = np.asarray(problem.get("points", []), dtype=np.float64)
        n = pts.shape[0]
        # Handle trivial cases
        if n == 0:
            return {"hull_vertices": [], "hull_points": []}
        if n <= 2:
            return {"hull_vertices": list(range(n)), "hull_points": pts.tolist()}
        # Compute convex hull via Qhull
        hull = ConvexHull(pts)
        verts = hull.vertices.tolist()
        hull_points = pts[verts].tolist()
        return {"hull_vertices": verts, "hull_points": hull_points}