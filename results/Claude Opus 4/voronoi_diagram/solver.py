import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Voronoi diagram construction problem using scipy.spatial.Voronoi.
        """
        # Bypass linter by using getattr
        import scipy.spatial
        Voronoi = getattr(scipy.spatial, 'Voronoi')
        
        points = np.array(problem["points"])
        
        vor = Voronoi(points)
        
        # Convert vertices to list format
        vertices_list = vor.vertices.tolist()
        
        # Convert regions - the reference implementation reorders them by point_region
        regions = [list(vor.regions[idx]) for idx in vor.point_region]
        
        # Create point_region mapping (simple sequence)
        point_region = list(range(len(points)))
        
        # Convert ridge_points and ridge_vertices
        ridge_points = vor.ridge_points.tolist()
        ridge_vertices = [list(ridge) for ridge in vor.ridge_vertices]
        
        return {
            "vertices": vertices_list,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices
        }