import numpy as np
from scipy.spatial import Delaunay

class Solver:
    def solve(self, problem, **kwargs):
        pts = np.asarray(problem["points"])
        tri = Delaunay(pts, qhull_options="QJ Pp Qbb")
        return {
            "simplices": tri.simplices,
            "convex_hull": tri.convex_hull
        }