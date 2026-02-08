from typing import Any
import numpy as np
from numba import njit


@njit(cache=True)
def _hull_ccw(px, py, order):
    n = len(order)
    if n <= 2:
        r = np.empty(n, dtype=np.int64)
        for i in range(n):
            r[i] = order[i]
        return r
    h = np.empty(2 * n, dtype=np.int64)
    k = 0
    for i in range(n):
        j = order[i]
        xj, yj = px[j], py[j]
        while k >= 2:
            o, a = h[k-2], h[k-1]
            if (px[a]-px[o])*(yj-py[o]) - (py[a]-py[o])*(xj-px[o]) <= 0.:
                k -= 1
            else:
                break
        h[k] = j
        k += 1
    t = k + 1
    for i in range(n-2, -1, -1):
        j = order[i]
        xj, yj = px[j], py[j]
        while k >= t:
            o, a = h[k-2], h[k-1]
            if (px[a]-px[o])*(yj-py[o]) - (py[a]-py[o])*(xj-px[o]) <= 0.:
                k -= 1
            else:
                break
        h[k] = j
        k += 1
    return h[:k-1]


@njit(cache=True)
def _full_hull(px, py):
    """Sort + hull in one call to avoid Python overhead."""
    n = len(px)
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        order[i] = i
    # Simple sort by (x, y) using merge sort approach
    # Use numpy argsort equivalent - actually just do insertion sort for moderate n
    # For large n this is slow, but we only call this for n <= 500
    # Actually let's just build indices and sort them
    # We'll use a simple O(n log n) sort
    _merge_sort(px, py, order, 0, n)
    return _hull_ccw(px, py, order)


@njit(cache=True)
def _merge_sort(px, py, arr, lo, hi):
    if hi - lo <= 1:
        return
    if hi - lo <= 16:
        # Insertion sort for small subarrays
        for i in range(lo + 1, hi):
            key = arr[i]
            kx, ky = px[key], py[key]
            j = i - 1
            while j >= lo:
                jv = arr[j]
                if px[jv] > kx or (px[jv] == kx and py[jv] > ky):
                    arr[j + 1] = arr[j]
                    j -= 1
                else:
                    break
            arr[j + 1] = key
        return
    mid = (lo + hi) // 2
    _merge_sort(px, py, arr, lo, mid)
    _merge_sort(px, py, arr, mid, hi)
    # Merge
    temp = np.empty(hi - lo, dtype=np.int64)
    i, j, k = lo, mid, 0
    while i < mid and j < hi:
        iv, jv = arr[i], arr[j]
        if px[iv] < px[jv] or (px[iv] == px[jv] and py[iv] <= py[jv]):
            temp[k] = iv
            i += 1
        else:
            temp[k] = jv
            j += 1
        k += 1
    while i < mid:
        temp[k] = arr[i]
        i += 1
        k += 1
    while j < hi:
        temp[k] = arr[j]
        j += 1
        k += 1
    for i in range(hi - lo):
        arr[lo + i] = temp[i]


@njit(cache=True)
def _filter_interior(px, py, hull_idx):
    n = len(px)
    nh = len(hull_idx)
    hx = np.empty(nh)
    hy = np.empty(nh)
    edx = np.empty(nh)
    edy = np.empty(nh)
    for j in range(nh):
        hx[j] = px[hull_idx[j]]
        hy[j] = py[hull_idx[j]]
    for j in range(nh):
        nj = (j + 1) % nh
        edx[j] = hx[nj] - hx[j]
        edy[j] = hy[nj] - hy[j]
    result = np.empty(n, dtype=np.int64)
    cnt = 0
    for i in range(n):
        x, y = px[i], py[i]
        inside = True
        for j in range(nh):
            if edx[j] * (y - hy[j]) - edy[j] * (x - hx[j]) <= 1e-10:
                inside = False
                break
        if not inside:
            result[cnt] = i
            cnt += 1
    return result[:cnt]


@njit(cache=True)
def _filter_and_hull(px, py, hull_idx):
    """Filter + sort + hull in one call."""
    n = len(px)
    nh = len(hull_idx)
    hx = np.empty(nh)
    hy = np.empty(nh)
    edx = np.empty(nh)
    edy = np.empty(nh)
    for j in range(nh):
        hx[j] = px[hull_idx[j]]
        hy[j] = py[hull_idx[j]]
    for j in range(nh):
        nj = (j + 1) % nh
        edx[j] = hx[nj] - hx[j]
        edy[j] = hy[nj] - hy[j]
    
    cand = np.empty(n, dtype=np.int64)
    cnt = 0
    for i in range(n):
        x, y = px[i], py[i]
        inside = True
        for j in range(nh):
            if edx[j] * (y - hy[j]) - edy[j] * (x - hx[j]) <= 1e-10:
                inside = False
                break
        if not inside:
            cand[cnt] = i
            cnt += 1
    
    if cnt < 3:
        return np.empty(0, dtype=np.int64), cnt
    
    # Sort candidates
    order = cand[:cnt].copy()
    _merge_sort(px, py, order, 0, cnt)
    
    return _hull_ccw(px, py, order), cnt


# Force JIT compilation
_px0 = np.array([0., 1., 0., 1., .5], dtype=np.float64)
_py0 = np.array([0., 0., 1., 1., .5], dtype=np.float64)
_h0 = _full_hull(_px0, _py0)
_filter_interior(_px0, _py0, _h0)
_filter_and_hull(_px0, _py0, _h0)


def _py_hull_small(pts, n):
    """Pure Python monotone chain for very small inputs."""
    indexed = sorted(range(n), key=lambda i: (pts[i][0], pts[i][1]))
    hull = []
    for i in range(n):
        idx = indexed[i]
        xi = pts[idx][0]
        yi = pts[idx][1]
        while len(hull) >= 2:
            o = hull[-2]
            a = hull[-1]
            if (pts[a][0] - pts[o][0]) * (yi - pts[o][1]) - (pts[a][1] - pts[o][1]) * (xi - pts[o][0]) <= 0:
                hull.pop()
            else:
                break
        hull.append(idx)
    t = len(hull) + 1
    for i in range(n - 2, -1, -1):
        idx = indexed[i]
        xi = pts[idx][0]
        yi = pts[idx][1]
        while len(hull) >= t:
            o = hull[-2]
            a = hull[-1]
            if (pts[a][0] - pts[o][0]) * (yi - pts[o][1]) - (pts[a][1] - pts[o][1]) * (xi - pts[o][0]) <= 0:
                hull.pop()
            else:
                break
        hull.append(idx)
    hull.pop()
    return hull


# Precompute trig values
_ND = 64
_ANG = np.linspace(0, 2 * np.pi, _ND, endpoint=False)
_COS_A = np.cos(_ANG)
_SIN_A = np.sin(_ANG)


class Solver:
    def solve(self, problem, **kwargs) -> Any:
        raw_points = problem["points"]
        n = len(raw_points)
        
        if n < 3:
            if isinstance(raw_points, np.ndarray):
                return {"hull_vertices": list(range(n)), "hull_points": raw_points.tolist()}
            return {"hull_vertices": list(range(n)), "hull_points": [list(p) for p in raw_points[:n]]}
        
        # For very small inputs, pure Python
        if n <= 80:
            if isinstance(raw_points, np.ndarray):
                pts_list = raw_points.tolist()
            elif isinstance(raw_points, list):
                pts_list = raw_points
            else:
                pts_list = [list(p) for p in raw_points]
            hi = _py_hull_small(pts_list, n)
            hp = [pts_list[i] for i in hi]
            return {"hull_vertices": hi, "hull_points": hp}
        
        pts = np.asarray(raw_points, dtype=np.float64)
        px = np.ascontiguousarray(pts[:, 0])
        py = np.ascontiguousarray(pts[:, 1])
        
        if n > 500:
            # Prefilter with extreme points
            proj = _COS_A[:, None] * px[None, :] + _SIN_A[:, None] * py[None, :]
            ei = np.unique(np.argmax(proj, axis=1))
            if len(ei) >= 3:
                eo = np.lexsort((py[ei], px[ei]))
                se = ei[eo].astype(np.int64)
                mh = _hull_ccw(px, py, se)
                if len(mh) >= 3:
                    hi, cnt = _filter_and_hull(px, py, mh)
                    if len(hi) >= 3 and cnt < n * 0.85:
                        return {"hull_vertices": hi.tolist(), "hull_points": pts[hi].tolist()}
        
        # Full numba computation
        hi = _full_hull(px, py)
        return {"hull_vertices": hi.tolist(), "hull_points": pts[hi].tolist()}