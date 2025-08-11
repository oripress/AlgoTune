import numpy as np
from scipy.spatial import ConvexHull
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Convex Hull problem using scipy.spatial.ConvexHull.
        """
        points = np.asarray(problem["points"], dtype=np.float64)
        
        n = len(points)
        if n < 3:
            hull_vertices = list(range(n))
            hull_points = points.tolist()
        else:
            # Use scipy's optimized ConvexHull
            hull = ConvexHull(points)
            
            # Get the vertices in counter-clockwise order
            hull_vertices = hull.vertices.tolist()
            hull_points = points[hull.vertices].tolist()
        
        return {
            "hull_vertices": hull_vertices,
            "hull_points": hull_points
        }