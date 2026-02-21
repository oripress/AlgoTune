from typing import Any
from scipy.spatial import Delaunay
import scipy.spatial._qhull as qhull

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        print(dir(qhull.Delaunay))
        tri = Delaunay(problem["points"], qhull_options="QJ")
        return {
            "simplices": tri.simplices,
            "convex_hull": tri.convex_hull,
        }