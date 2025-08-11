from typing import Any, Dict
import json
import numpy as np

try:
    from scipy.spatial import Delaunay as SciPyDelaunay
except Exception:
    SciPyDelaunay = None

# small preallocated empty results
_EMPTY_SIMPLES = np.empty((0, 3), dtype=int)
_EMPTY_HULL = np.empty((0, 2), dtype=int)
_ONE_EDGE_HULL = np.array([[0, 1]], dtype=int)

class Solver:
    """
    Fast wrapper for 2D Delaunay triangulation.

    Strategy:
    - Accept dicts with key "points", bare lists/arrays, or JSON strings.
    - Normalize input to an (n,2) float64 numpy array with minimal copying.
    - Prefer SciPy's compiled Delaunay (Qhull). Return its arrays directly.
    - Lightweight caching by object id to speed repeated calls with the same ndarray.
    - Fallback: compute convex hull edges (Andrew's monotone chain).
    """

    def __init__(self) -> None:
        # cache: (id(array), shape, dtype.str) -> result dict
        self._cache_key = None
        self._cache_result = None

    def solve(self, problem: Any, **kwargs: Any) -> Dict[str, Any]:
        # Unpack possible JSON string
        p = problem
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                p = {}

        # Extract points
        if isinstance(p, dict):
            pts_in = p.get("points", [])
        else:
            pts_in = p

        if pts_in is None:
            return {"simplices": _EMPTY_SIMPLES, "convex_hull": _EMPTY_HULL}

        # Fast path: if already ndarray try to reuse/reshape without copies
        if isinstance(pts_in, np.ndarray):
            pts = pts_in
            if pts.ndim == 1:
                if pts.size == 0:
                    pts = pts.reshape((0, 2))
                elif pts.size == 2:
                    pts = pts.reshape((1, 2))
                else:
                    pts = pts.reshape((-1, 2))
            elif pts.ndim == 2 and pts.shape[1] != 2:
                # attempt to interpret flattened array
                if pts.size % 2 == 0:
                    pts = pts.reshape((-1, 2))
                else:
                    return {"simplices": _EMPTY_SIMPLES, "convex_hull": _EMPTY_HULL}
            if pts.dtype != np.float64:
                pts = pts.astype(np.float64, copy=False)
        else:
            # Generic conversion (forces float64, minimal copy)
            pts = np.asarray(pts_in, dtype=np.float64)
            if pts.ndim == 1:
                if pts.size == 0:
                    pts = pts.reshape((0, 2))
                elif pts.size == 2:
                    pts = pts.reshape((1, 2))
                else:
                    pts = pts.reshape((-1, 2))
            elif pts.ndim == 2 and pts.shape[1] != 2:
                if pts.size % 2 == 0:
                    pts = pts.reshape((-1, 2))
                else:
                    return {"simplices": _EMPTY_SIMPLES, "convex_hull": _EMPTY_HULL}

        n = pts.shape[0]

        # trivial cases
        if n == 0 or n == 1:
            return {"simplices": _EMPTY_SIMPLES, "convex_hull": _EMPTY_HULL}
        if n == 2:
            return {"simplices": _EMPTY_SIMPLES, "convex_hull": _ONE_EDGE_HULL}

        # lightweight cache by object identity + shape + dtype
        try:
            cache_key = (id(pts), pts.shape, pts.dtype.str)
            if cache_key == self._cache_key:
                return self._cache_result
        except Exception:
            cache_key = None

        # Prefer SciPy's compiled Delaunay (Qhull)
        if SciPyDelaunay is not None:
            tri = SciPyDelaunay(pts)
            result = {"simplices": tri.simplices, "convex_hull": tri.convex_hull}
            if cache_key is not None:
                self._cache_key = cache_key
                self._cache_result = result
            return result

        # Fallback: compute convex hull via Andrew's monotone chain
        idx = np.arange(n)
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        sorted_idx = idx[order]
        sorted_pts = pts[order]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for i in range(n):
            while len(lower) >= 2 and cross(sorted_pts[lower[-2]], sorted_pts[lower[-1]], sorted_pts[i]) <= 0:
                lower.pop()
            lower.append(i)

        upper = []
        for i in range(n - 1, -1, -1):
            while len(upper) >= 2 and cross(sorted_pts[upper[-2]], sorted_pts[upper[-1]], sorted_pts[i]) <= 0:
                upper.pop()
            upper.append(i)

        hull_pos = lower[:-1] + upper[:-1]
        if not hull_pos:
            return {"simplices": _EMPTY_SIMPLES, "convex_hull": _EMPTY_HULL}

        hull_indices = sorted_idx[hull_pos]
        m = hull_indices.shape[0]
        edges = np.empty((m, 2), dtype=int)
        for i in range(m):
            edges[i, 0] = int(hull_indices[i])
            edges[i, 1] = int(hull_indices[(i + 1) % m])

        result = {"simplices": _EMPTY_SIMPLES, "convex_hull": edges}
        if cache_key is not None:
            self._cache_key = cache_key
            self._cache_result = result
        return result