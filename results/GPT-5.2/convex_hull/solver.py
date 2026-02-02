from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

def _monotone_chain_ccw_indices(pts: np.ndarray) -> np.ndarray:
    """Convex hull (CCW) of a small point set; returns indices into pts (no repeated start)."""
    n = int(pts.shape[0])
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    st: list[int] = []

    def cross(i1: int, i2: int, i3: int) -> float:
        a = pts[i1]
        b = pts[i2]
        c = pts[i3]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for k in range(n):
        i = int(order[k])
        while len(st) >= 2 and cross(st[-2], st[-1], i) <= 0.0:
            st.pop()
        st.append(i)

    t = len(st)
    for k in range(n - 2, -1, -1):
        i = int(order[k])
        while len(st) >= t + 1 and cross(st[-2], st[-1], i) <= 0.0:
            st.pop()
        st.append(i)

    st.pop()
    return np.asarray(st, dtype=np.int64)

if njit is not None:

    @njit(cache=True, fastmath=True)
    def _extreme8_idx(points: np.ndarray) -> np.ndarray:
        """Return 8 extreme-point indices: min/max x,y and min/max (x+y),(x-y)."""
        n = points.shape[0]
        # initialize with 0
        minx = maxx = miny = maxy = mins1 = maxs1 = mins2 = maxs2 = 0
        x0 = points[0, 0]
        y0 = points[0, 1]
        minxv = maxxv = x0
        minyv = maxyv = y0
        s10 = x0 + y0
        s20 = x0 - y0
        mins1v = maxs1v = s10
        mins2v = maxs2v = s20

        for i in range(1, n):
            x = points[i, 0]
            y = points[i, 1]
            if x < minxv:
                minxv = x
                minx = i
            if x > maxxv:
                maxxv = x
                maxx = i
            if y < minyv:
                minyv = y
                miny = i
            if y > maxyv:
                maxyv = y
                maxy = i
            s1 = x + y
            if s1 < mins1v:
                mins1v = s1
                mins1 = i
            if s1 > maxs1v:
                maxs1v = s1
                maxs1 = i
            s2 = x - y
            if s2 < mins2v:
                mins2v = s2
                mins2 = i
            if s2 > maxs2v:
                maxs2v = s2
                maxs2 = i

        out = np.empty(8, dtype=np.int64)
        out[0] = minx
        out[1] = maxx
        out[2] = miny
        out[3] = maxy
        out[4] = mins1
        out[5] = maxs1
        out[6] = mins2
        out[7] = maxs2
        return out

    @njit(cache=True, fastmath=True)
    def _keep_outside_or_on_poly(points: np.ndarray, poly: np.ndarray, eps: float) -> np.ndarray:
        """
        Return keep mask where keep=True for points outside or on the CCW convex polygon `poly`,
        and keep=False for points strictly inside (min cross > eps).
        """
        n = points.shape[0]
        m = poly.shape[0]
        keep = np.empty(n, dtype=np.bool_)
        for j in range(n):
            px = points[j, 0]
            py = points[j, 1]
            minc = 1.0e308
            for i in range(m):
                x1 = poly[i, 0]
                y1 = poly[i, 1]
                i2 = i + 1
                if i2 == m:
                    i2 = 0
                x2 = poly[i2, 0]
                y2 = poly[i2, 1]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if cross < minc:
                    minc = cross
            keep[j] = minc <= eps
        return keep

def _unique_small_ints(a: np.ndarray) -> np.ndarray:
    """Unique for very small 1D int arrays (len<=8), preserving order of first occurrence."""
    out = []
    seen = set()
    for v in a.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return np.array(out, dtype=np.int64)

@dataclass
class Solver:
    def __post_init__(self) -> None:
        # Compile numba kernels outside timed solve().
        if njit is not None:
            pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.2]], dtype=np.float64)
            poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
            _ = _extreme8_idx(pts)
            _ = _keep_outside_or_on_poly(pts, poly, 1e-9)

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        pts = np.asarray(problem["points"], dtype=np.float64)
        n = int(pts.shape[0])
        if n == 0:
            return {"hull_vertices": np.empty(0, dtype=np.int64), "hull_points": pts}
        if n == 1:
            hv = np.array([0], dtype=np.int64)
            return {"hull_vertices": hv, "hull_points": pts[hv]}

        # For small n, the filtering overhead often isn't worth it.
        if n < 2048:
            hull = ConvexHull(pts)
            hv = hull.vertices.astype(np.int64, copy=False)
            return {"hull_vertices": hv, "hull_points": pts[hv]}

        # 8-direction extremes (Akl–Toussaint heuristic), computed without extra passes/allocs.
        if njit is not None:
            ext_idx_raw = _extreme8_idx(pts)
            ext_idx = _unique_small_ints(ext_idx_raw)
        else:  # pragma: no cover
            x = pts[:, 0]
            y = pts[:, 1]
            s1 = x + y
            s2 = x - y
            ext_idx = np.array(
                [
                    int(np.argmin(x)),
                    int(np.argmax(x)),
                    int(np.argmin(y)),
                    int(np.argmax(y)),
                    int(np.argmin(s1)),
                    int(np.argmax(s1)),
                    int(np.argmin(s2)),
                    int(np.argmax(s2)),
                ],
                dtype=np.int64,
            )
            ext_idx = np.unique(ext_idx)

        if ext_idx.size < 3:
            hull = ConvexHull(pts)
            hv = hull.vertices.astype(np.int64, copy=False)
            return {"hull_vertices": hv, "hull_points": pts[hv]}

        ext_pts = pts[ext_idx]
        poly_pos = _monotone_chain_ccw_indices(ext_pts)
        if poly_pos.size < 3:
            hull = ConvexHull(pts)
            hv = hull.vertices.astype(np.int64, copy=False)
            return {"hull_vertices": hv, "hull_points": pts[hv]}

        poly = ext_pts[poly_pos]
        eps = 1e-9

        if njit is not None:
            keep = _keep_outside_or_on_poly(pts, poly, eps)
        else:  # pragma: no cover
            # Vectorized fallback (allocates temporaries).
            x = pts[:, 0]
            y = pts[:, 1]
            m = int(poly.shape[0])
            min_cross = np.full(n, np.inf, dtype=np.float64)
            for i in range(m):
                p1 = poly[i]
                p2 = poly[(i + 1) % m]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                c = dx * (y - p1[1]) - dy * (x - p1[0])
                np.minimum(min_cross, c, out=min_cross)
            keep = min_cross <= eps

        keep[ext_idx] = True
        kept_idx = np.flatnonzero(keep)

        # If reduction is tiny, skip reduced hull.
        if kept_idx.size > (n * 0.95):
            hull = ConvexHull(pts)
            hv = hull.vertices.astype(np.int64, copy=False)
            return {"hull_vertices": hv, "hull_points": pts[hv]}

        pts_r = pts[kept_idx]
        hull = ConvexHull(pts_r)
        hv = kept_idx[hull.vertices].astype(np.int64, copy=False)
        return {"hull_vertices": hv, "hull_points": pts[hv]}