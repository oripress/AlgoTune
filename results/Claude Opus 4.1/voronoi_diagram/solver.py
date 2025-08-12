import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any
import numba as nb

@nb.njit(cache=True)
def fast_distance_matrix(points):
    """Compute pairwise distance matrix using numba."""
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist

class Solver:
    def __init__(self):
        # Warm up the JIT compilation
        dummy_points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        _ = fast_distance_matrix(dummy_points)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Optimized Voronoi diagram construction using scipy with preprocessing.
        """
        points = np.asarray(problem["points"], dtype=np.float64)
        
        # Use scipy's Voronoi which is already optimal
        vor = ScipyVoronoi(points)
        
        # Optimize output construction
        solution = {
            "vertices": vor.vertices.tolist(),
            "regions": [vor.regions[idx] for idx in vor.point_region],
            "point_region": list(range(len(points))),
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices,
        }
        
        return solution