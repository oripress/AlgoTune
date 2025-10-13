from typing import Any, Dict

from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Construct a Voronoi diagram using scipy.spatial.Voronoi and format the result
        with minimal overhead. Bounds are ignored to match the reference behavior.
        """
        points = problem["points"]

        vor = ScipyVoronoi(points)

        return {
            "vertices": vor.vertices,               # ndarray (n_vertices, 2)
            "regions": vor.regions,                 # list of lists of vertex indices
            "point_region": vor.point_region,       # ndarray (n_points,)
            "ridge_points": vor.ridge_points,       # ndarray (n_ridges, 2)
            "ridge_vertices": vor.ridge_vertices,   # list of [v1, v2], -1 allowed
        }