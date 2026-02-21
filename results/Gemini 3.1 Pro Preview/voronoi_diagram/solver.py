import numpy as np
from scipy.spatial._qhull import _Qhull

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        points = np.asarray(problem["points"], dtype=np.float64)
        q = _Qhull(b"v", points, b"Qbb Qc Qz", b"", b"")
        vertices, ridge_points, ridge_vertices, regions, point_region = q.get_voronoi_diagram()
        
        return {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }