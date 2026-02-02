from typing import Any
import numpy as np
from scipy.spatial._qhull import _Qhull

class Solver:
    __slots__ = ()
    
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Voronoi diagram construction problem using low-level qhull.
        """
        points = np.asarray(problem["points"], dtype=np.float64)
        n = len(points)
        
        # Use _Qhull directly (what scipy.Voronoi does internally)
        qhull = _Qhull(b"v", points, options=b"Qbb Qc Qz", 
                       required_options=b"Qz", incremental=False)
        
        # Get Voronoi diagram components directly
        vertices, ridge_points, ridge_vertices, regions, point_region = qhull.get_voronoi_diagram()
        
        qhull.close()
        
        # Reorder regions by point_region
        regions_reordered = [regions[idx] for idx in point_region]
        
        return {
            "vertices": vertices,
            "regions": regions_reordered,
            "point_region": np.arange(n, dtype=np.intp),
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }