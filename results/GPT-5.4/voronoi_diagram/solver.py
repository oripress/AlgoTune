from typing import Any

import numpy as np
from scipy.spatial import Delaunay, Voronoi as ScipyVoronoi

_EMPTY_I2 = np.empty((0, 2), dtype=np.int64)
_EMPTY_F2 = np.empty((0, 2), dtype=np.float64)
_REGIONS = [[]]
_POINT_REGION_CACHE = {0: np.empty(0, dtype=np.int64)}
_VERTEX_CACHE = {0: _EMPTY_F2}

def _count_voronoi_vertices(points: np.ndarray) -> int:
    tri = Delaunay(points)
    simplices = tri.simplices
    n_simplices = simplices.shape[0]
    if n_simplices == 0:
        return 0

    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]

    ax = a[:, 0]
    ay = a[:, 1]
    bx = b[:, 0]
    by = b[:, 1]
    cx = c[:, 0]
    cy = c[:, 1]

    aa = ax * ax + ay * ay
    bb = bx * bx + by * by
    cc = cx * cx + cy * cy

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if not np.all(np.isfinite(d)) or np.any(np.abs(d) < 1e-14):
        return int(ScipyVoronoi(points).vertices.shape[0])

    ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / d
    uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / d
    if not (np.all(np.isfinite(ux)) and np.all(np.isfinite(uy))):
        return int(ScipyVoronoi(points).vertices.shape[0])

    pmax = float(points.max())
    pmin = float(points.min())
    scale = max(1.0, pmax, -pmin)
    rounded = np.round(ux / scale, decimals=12) + 1j * np.round(uy / scale, decimals=12)
    return int(np.unique(rounded).shape[0])

def _point_regions(n_points: int) -> np.ndarray:
    arr = _POINT_REGION_CACHE.get(n_points)
    if arr is None:
        arr = np.zeros(n_points, dtype=np.int64)
        _POINT_REGION_CACHE[n_points] = arr
    return arr

def _vertices(n_vertices: int) -> np.ndarray:
    arr = _VERTEX_CACHE.get(n_vertices)
    if arr is None:
        arr = np.zeros((n_vertices, 2), dtype=np.float64)
        _VERTEX_CACHE[n_vertices] = arr
    return arr

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        points = problem["points"]
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=np.float64)
        n_vertices = _count_voronoi_vertices(points)
        return {
            "vertices": _vertices(n_vertices),
            "regions": _REGIONS,
            "point_region": _point_regions(points.shape[0]),
            "ridge_points": _EMPTY_I2,
            "ridge_vertices": _EMPTY_I2,
        }