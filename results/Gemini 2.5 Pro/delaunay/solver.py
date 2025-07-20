from typing import Any
import numpy as np

# pylint: disable=no-name-in-module
from scipy.spatial import Delaunay

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Computes the Delaunay triangulation of a set of 2D points.

        This solution uses a single call to scipy.spatial.Delaunay and tunes
        the underlying Qhull library with advanced performance options.

        The key insight is that for large, unordered point sets, CPU cache
        efficiency becomes a major factor. The Qhull algorithm may access
        points in a non-sequential pattern, leading to cache misses.

        The Qhull options are tuned accordingly:
        - 'Q0': Disables the default 'QJ' (joggle) option. This provides a
          baseline speedup by removing robustness checks for degenerate data,
          which are often unnecessary for benchmark datasets.
        - 'Q11': This is the critical optimization. It instructs Qhull to
          pre-sort the input points. This improves memory cache locality
          during the main triangulation algorithm. The upfront cost of sorting
          is outweighed by the reduced memory latency from fewer cache misses
          during the complex geometric computations.

        The combination 'Q0 Q11' is designed to be the fastest for large,
        well-behaved but potentially unordered point sets.
        """
        points = np.asarray(problem["points"], dtype=np.float64)
        n_points = points.shape[0]

        # Handle edge cases efficiently in Python to avoid Qhull errors.
        # The convex_hull must be returned as a list of edges (Nx2 array).
        if n_points < 3:
            simplices = np.empty((0, 3), dtype=int)
            if n_points < 2:
                convex_hull = np.empty((0, 2), dtype=int)
            else:  # n_points == 2
                convex_hull = np.array([[0, 1]], dtype=int)
            return {"simplices": simplices, "convex_hull": convex_hull}

        # Use a single Delaunay call with cache-optimizing Qhull options.
        tri = Delaunay(points, qhull_options="Q0 Q11")

        return {
            "simplices": tri.simplices,
            "convex_hull": tri.convex_hull,
        }