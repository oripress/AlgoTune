import numpy as np
from typing import Any
import hull

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points_list = problem["points"]
        # Ensure points are float64 and contiguous for Cython memoryview
        points = np.array(points_list, dtype=np.float64)
        
        # Usually np.array() creates a contiguous array, but to be safe and satisfy type checkers/linters
        if not points.flags.c_contiguous:
            points = np.ascontiguousarray(points)
        
        hull_vertices, hull_points = hull.compute_hull(points)
        
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}