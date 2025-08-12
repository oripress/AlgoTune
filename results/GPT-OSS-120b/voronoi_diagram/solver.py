import numpy as np
from typing import Any, Dict
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float64)
        problem["points"] = points
        vor = ScipyVoronoi(points)
        # Directly use scipy's structures without extra copying for speed
        return {
            "vertices": vor.vertices,
            "regions": vor.regions,
            "point_region": vor.point_region,
            "ridge_points": vor.ridge_points,
            "ridge_vertices": vor.ridge_vertices,
        }