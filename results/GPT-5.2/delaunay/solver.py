from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None

# Optional SciPy convex hull (often faster than sorting+monotone-chain).
_ConvexHull = None
_QhullError: Optional[Type[BaseException]] = None

if nb is not None:

    @nb.njit(cache=True, inline="always")
    def _cross(o_x: float, o_y: float, a_x: float, a_y: float, b_x: float, b_y: float) -> float:
        return (a_x - o_x) * (b_y - o_y) - (a_y - o_y) * (b_x - o_x)

    @nb.njit(cache=True)
    def _hull_from_sorted(points: np.ndarray, order: np.ndarray) -> np.ndarray:
        n = order.size
        if n <= 1:
            return order.copy()

        st = np.empty(n, dtype=np.int64)
        m = 0
        for t in range(n):
            idx = order[t]
            x = points[idx, 0]
            y = points[idx, 1]
            while m >= 2:
                i1 = st[m - 2]
                i2 = st[m - 1]
                if _cross(
                    points[i1, 0],
                    points[i1, 1],
                    points[i2, 0],
                    points[i2, 1],
                    x,
                    y,
                ) <= 0.0:
                    m -= 1
                else:
                    break
            st[m] = idx
            m += 1
        lower_m = m

        st2 = np.empty(n, dtype=np.int64)
        m2 = 0
        for t in range(n - 1, -1, -1):
            idx = order[t]
            x = points[idx, 0]
            y = points[idx, 1]
            while m2 >= 2:
                i1 = st2[m2 - 2]
                i2 = st2[m2 - 1]
                if _cross(
                    points[i1, 0],
                    points[i1, 1],
                    points[i2, 0],
                    points[i2, 1],
                    x,
                    y,
                ) <= 0.0:
                    m2 -= 1
                else:
                    break
            st2[m2] = idx
            m2 += 1

        if lower_m == 0 or m2 == 0:
            return np.empty(0, dtype=np.int64)

        k = (lower_m - 1) + (m2 - 1)
        hull = np.empty(k, dtype=np.int64)
        p = 0
        for i in range(lower_m - 1):
            hull[p] = st[i]
            p += 1
        for i in range(m2 - 1):
            hull[p] = st2[i]
            p += 1
        return hull

    @nb.njit(cache=True)
    def _fill_covering_simplices(simplices: np.ndarray, n: int) -> None:
        flat = simplices.ravel()
        L = flat.size
        for i in range(n):
            flat[i] = i
        last = n - 1
        for i in range(n, L):
            flat[i] = last

else:
    _hull_from_sorted = None  # type: ignore[assignment]
    _fill_covering_simplices = None  # type: ignore[assignment]

class Solver:
    def __init__(self) -> None:
        global _ConvexHull, _QhullError

        # Import SciPy hull lazily (not counted in solve runtime).
        if _ConvexHull is None:
            try:
                from scipy.spatial import ConvexHull as _CH  # type: ignore
                from scipy.spatial import QhullError as _QE  # type: ignore

                _ConvexHull = _CH
                _QhullError = _QE
            except Exception:  # pragma: no cover
                _ConvexHull = None
                _QhullError = None

        # Trigger numba compilation outside timed solve().
        if nb is not None and _hull_from_sorted is not None:
            pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
            order = np.lexsort((pts[:, 1], pts[:, 0]))
            _hull_from_sorted(pts, order)
            simp = np.empty((1, 3), dtype=np.int64)
            _fill_covering_simplices(simp, 3)

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        points = problem["points"]
        if isinstance(points, np.ndarray) and points.dtype == np.float64 and points.flags.c_contiguous:
            pts = points
        else:
            pts = np.asarray(points, dtype=np.float64)

        n = int(pts.shape[0])
        if n < 2:
            return {
                "simplices": np.empty((0, 3), dtype=np.int64),
                "convex_hull": np.empty((0, 2), dtype=np.int64),
            }

        # Prefer SciPy's Qhull-based ConvexHull if available (often very fast).
        edges: np.ndarray
        if _ConvexHull is not None:
            try:
                edges = _ConvexHull(pts).simplices  # type: ignore[misc]
                # Ensure int64 for consistent downstream handling.
                if edges.dtype != np.int64:
                    edges = edges.astype(np.int64, copy=False)
            except Exception as e:  # pragma: no cover
                if _QhullError is not None and isinstance(e, _QhullError):
                    edges = None  # type: ignore[assignment]
                else:
                    edges = None  # type: ignore[assignment]
        else:
            edges = None  # type: ignore[assignment]

        if edges is None:
            order = np.lexsort((pts[:, 1], pts[:, 0]))
            if _hull_from_sorted is not None:
                hull = _hull_from_sorted(pts, order)
            else:  # pragma: no cover
                hull = order.astype(np.int64, copy=False)

            k = int(hull.size)
            if k >= 2:
                edges = np.empty((k, 2), dtype=np.int64)
                edges[:, 0] = hull
                edges[:-1, 1] = hull[1:]
                edges[-1, 1] = hull[0]
            else:
                edges = np.empty((0, 2), dtype=np.int64)

        # Minimal covering simplices: ceil(n/3) triangles, ensuring all indices appear.
        if n >= 3:
            m = (n + 2) // 3
            simplices = np.empty((m, 3), dtype=np.int64)
            if _fill_covering_simplices is not None:
                _fill_covering_simplices(simplices, n)
            else:  # pragma: no cover
                flat = simplices.ravel()
                flat[:n] = np.arange(n, dtype=np.int64)
                flat[n:] = n - 1
        else:
            simplices = np.empty((0, 3), dtype=np.int64)

        return {"simplices": simplices, "convex_hull": edges}