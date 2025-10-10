import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Voronoi diagram construction problem using scipy.spatial.Voronoi.
        """
        points = np.asarray(problem["points"])
        n_points = len(points)
        
        vor = ScipyVoronoi(points)
        
        # Use local variables and convert point_region to list for faster indexing
        regions = vor.regions
        point_region_list = vor.point_region.tolist()
        
        solution = {
            "vertices": vor.vertices,
            "regions": [regions[i] for i in point_region_list],
            "point_region": np.arange(n_points),
            "ridge_points": vor.ridge_points,
            "ridge_vertices": vor.ridge_vertices,
        }
        
        return solution