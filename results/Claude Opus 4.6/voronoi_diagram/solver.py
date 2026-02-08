import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        points = problem["points"]
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=np.float64)
        
        n = len(points)
        
        vor = ScipyVoronoi(points)
        
        # Reorder regions by point_region mapping using list comprehension with direct access
        all_regions = vor.regions
        pr = vor.point_region
        reordered_regions = [all_regions[pr[i]] for i in range(n)]
        
        return {
            "vertices": vor.vertices,
            "regions": reordered_regions,
            "point_region": np.arange(n, dtype=np.intp),
            "ridge_points": vor.ridge_points,
            "ridge_vertices": vor.ridge_vertices,
        }