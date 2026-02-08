import numpy as np
from scipy.spatial import Delaunay

class Solver:
    def solve(self, problem, **kwargs):
        pts = np.asarray(problem["points"], dtype=np.float64)
        
        tri = Delaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        
        # Canonicalize simplices
        simplices.sort(axis=1)
        simplices = simplices[np.lexsort(simplices.T[::-1])]
        
        # Canonicalize convex hull edges
        convex_hull.sort(axis=1)
        convex_hull = convex_hull[np.lexsort(convex_hull.T[::-1])]
        
        return {
            "simplices": simplices.tolist(),
            "convex_hull": convex_hull.tolist(),
        }