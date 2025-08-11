import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Compute Delaunay triangulation with optimized memory layout."""
        try:
            from scipy.spatial import Delaunay
            # Convert to numpy array with optimal memory layout
            points = np.asarray(problem["points"], dtype=np.float64, order='C')
            tri = Delaunay(points)
            return {
                "simplices": tri.simplices,
                "convex_hull": tri.convex_hull
            }
        except ImportError:
            return {"simplices": [], "convex_hull": []}