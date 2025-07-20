from scipy.spatial.qhull import Delaunay as SciPyDelaunay
import numpy as np

# Pre-allocated empty outputs for trivial cases
_EMPTY_SIMPLICES = np.empty((0, 3), dtype=int)
_EMPTY_HULL = np.empty((0, 2), dtype=int)
_TWO_HULL = np.array([[0, 1]], dtype=int)

class Solver:
    def solve(self, problem, **kwargs):
        pts = np.asarray(problem.get("points", []), dtype=float)
        n = pts.shape[0]
        if n < 2:
            return {
                "simplices": _EMPTY_SIMPLICES,
                "convex_hull": _EMPTY_HULL,
            }
        if n == 2:
            return {
                "simplices": _EMPTY_SIMPLICES,
                "convex_hull": _TWO_HULL,
            }
        # Use Qhull randomization (Qx) and bounding box (Qbb) for faster average-case
        tri = SciPyDelaunay(pts, qhull_options="Qbb Qx")
        return {
            "simplices": tri.simplices,
            "convex_hull": tri.convex_hull,
        }