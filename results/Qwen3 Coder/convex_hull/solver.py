import numpy as np
from typing import Any
from scipy.spatial.qhull import ConvexHull

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Convex Hull problem using scipy.spatial.ConvexHull."""
        points = problem["points"]
        hull = ConvexHull(points)
        
        # Get the vertices of the convex hull
        hull_vertices = hull.vertices.tolist()
        
        # Get the points that form the hull in order
        hull_points = points[hull.vertices].tolist()
        
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}