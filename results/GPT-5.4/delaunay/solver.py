import math
from typing import Any

import numpy as np
from scipy.spatial import Delaunay as _Delaunay

_TRI3 = np.array([[0, 1, 2]], dtype=np.int64)
_HULL3 = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def _incircle(a, b, c, d) -> float:
    adx = a[0] - d[0]
    ady = a[1] - d[1]
    bdx = b[0] - d[0]
    bdy = b[1] - d[1]
    cdx = c[0] - d[0]
    cdy = c[1] - d[1]

    abdet = adx * bdy - bdx * ady
    bcdet = bdx * cdy - cdx * bdy
    cadet = cdx * ady - adx * cdy
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    det = alift * bcdet + blift * cadet + clift * abdet
    return det if _orient(a, b, c) > 0.0 else -det

def _solve4(points):
    eps = 1e-12
    idx = (0, 1, 2, 3)

    for inner in idx:
        outer = [j for j in idx if j != inner]
        a = points[outer[0]]
        b = points[outer[1]]
        c = points[outer[2]]
        if abs(_orient(a, b, c)) <= eps:
            continue
        p = points[inner]
        o1 = _orient(a, b, p)
        o2 = _orient(b, c, p)
        o3 = _orient(c, a, p)
        if (o1 >= -eps and o2 >= -eps and o3 >= -eps) or (
            o1 <= eps and o2 <= eps and o3 <= eps
        ):
            return {
                "simplices": np.array(
                    [
                        [inner, outer[0], outer[1]],
                        [inner, outer[1], outer[2]],
                        [inner, outer[2], outer[0]],
                    ],
                    dtype=np.int64,
                ),
                "convex_hull": np.array(
                    [
                        [outer[0], outer[1]],
                        [outer[1], outer[2]],
                        [outer[2], outer[0]],
                    ],
                    dtype=np.int64,
                ),
            }

    cx = 0.25 * sum(points[i][0] for i in idx)
    cy = 0.25 * sum(points[i][1] for i in idx)
    order = sorted(idx, key=lambda i: math.atan2(points[i][1] - cy, points[i][0] - cx))
    a, b, c, d = order

    if min(
        abs(_orient(points[a], points[b], points[c])),
        abs(_orient(points[b], points[c], points[d])),
    ) <= eps:
        return None

    hull = np.array([[a, b], [b, c], [c, d], [d, a]], dtype=np.int64)
    det = _incircle(points[a], points[b], points[c], points[d])
    if det <= eps:
        simplices = np.array([[a, b, c], [a, c, d]], dtype=np.int64)
    else:
        simplices = np.array([[a, b, d], [b, c, d]], dtype=np.int64)
    return {"simplices": simplices, "convex_hull": hull}

class Solver:
    __slots__ = ()

    def solve(self, problem, **kwargs) -> Any:
        pts = problem["points"]
        n = len(pts)

        if n == 3:
            return {"simplices": _TRI3, "convex_hull": _HULL3}

        if n == 4:
            ans = _solve4(pts)
            if ans is not None:
                return ans

        tri = _Delaunay(pts)
        return {"simplices": tri.simplices, "convex_hull": tri.convex_hull}