from typing import Any
import numpy as np
# pylint: disable=no-name-in-module
from scipy.spatial import Voronoi

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Computes the Voronoi diagram using a hybrid strategy.

        Optimization:
        The best stable performance was achieved with a hybrid strategy: try a
        fast, non-robust method and fall back to a slower, robust one. This
        solution aims to improve that strategy by optimizing the fast path.

        1. Fast Path: It first attempts computation with 'Qbb Qz'. The 'Qz'
           option adds a point at infinity, changing the underlying geometric
           computation from a paraboloid hull to a sphere hull. The hypothesis
           is that for this benchmark's data, this may be a faster path than
           the standard options, while still failing gracefully (with a
           catchable exception) on degenerate inputs that require 'Qc'.

        2. Robust Fallback: If the fast path fails, it falls back to the
           proven, stable 'Qbb Qc' options.

        This approach seeks to improve on the previous best speedup by finding
        a more efficient primary computation path.
        """
        points = np.asarray(problem["points"])

        try:
            # Fast path: Use 'Qz' to change the hull geometry, hoping it's
            # faster for the successful cases than 'Qbb' alone.
            vor = Voronoi(points, qhull_options="Qbb Qz")
        except Exception:
            # Fallback path: If the fast path fails, use the proven robust options.
            vor = Voronoi(points, qhull_options="Qbb Qc")

        # Construct the solution dictionary directly from the Voronoi object.
        solution = {
            "vertices": vor.vertices.tolist(),
            "regions": vor.regions,
            "point_region": vor.point_region.tolist(),
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices,
        }

        return solution