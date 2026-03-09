from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np
from scipy.spatial import ConvexHull

@nb.njit(cache=True, fastmath=True)
def _monotonic_chain(points: np.ndarray, order: np.ndarray) -> np.ndarray:
    n = order.shape[0]
    hull = np.empty(2 * n, dtype=np.int64)
    k = 0

    for t in range(n):
        idx = order[t]
        px = points[idx, 0]
        py = points[idx, 1]
        while k >= 2:
            a = hull[k - 2]
            b = hull[k - 1]
            ax = points[a, 0]
            ay = points[a, 1]
            bx = points[b, 0]
            by = points[b, 1]
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) <= 0.0:
                k -= 1
            else:
                break
        hull[k] = idx
        k += 1

    t0 = k + 1
    for t in range(n - 2, -1, -1):
        idx = order[t]
        px = points[idx, 0]
        py = points[idx, 1]
        while k >= t0:
            a = hull[k - 2]
            b = hull[k - 1]
            ax = points[a, 0]
            ay = points[a, 1]
            bx = points[b, 0]
            by = points[b, 1]
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) <= 0.0:
                k -= 1
            else:
                break
        hull[k] = idx
        k += 1

    return hull[: k - 1]

@nb.njit(cache=True, fastmath=True)
def _outside_inner_polygon_mask(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    m = poly.shape[0]
    keep = np.empty(n, dtype=np.bool_)

    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]
        outside = False
        for j in range(m):
            j2 = 0 if j + 1 == m else j + 1
            x1 = poly[j, 0]
            y1 = poly[j, 1]
            x2 = poly[j2, 0]
            y2 = poly[j2, 1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) < 0.0:
                outside = True
                break
        keep[i] = outside

    return keep

@nb.njit(cache=True, fastmath=True)
def _extreme_indices_8(points: np.ndarray) -> np.ndarray:
    min_x = max_x = points[0, 0]
    min_y = max_y = points[0, 1]
    s0 = points[0, 0] + points[0, 1]
    d0 = points[0, 0] - points[0, 1]
    min_s = max_s = s0
    min_d = max_d = d0

    idx_min_x = idx_max_x = 0
    idx_min_y = idx_max_y = 0
    idx_min_s = idx_max_s = 0
    idx_min_d = idx_max_d = 0

    for i in range(1, points.shape[0]):
        x = points[i, 0]
        y = points[i, 1]
        s = x + y
        d = x - y

        if x < min_x:
            min_x = x
            idx_min_x = i
        if x > max_x:
            max_x = x
            idx_max_x = i

        if y < min_y:
            min_y = y
            idx_min_y = i
        if y > max_y:
            max_y = y
            idx_max_y = i

        if s < min_s:
            min_s = s
            idx_min_s = i
        if s > max_s:
            max_s = s
            idx_max_s = i

        if d < min_d:
            min_d = d
            idx_min_d = i
        if d > max_d:
            max_d = d
            idx_max_d = i

    return np.array(
        [
            idx_min_x,
            idx_max_x,
            idx_min_y,
            idx_max_y,
            idx_min_s,
            idx_max_s,
            idx_min_d,
            idx_max_d,
        ],
        dtype=np.int64,
    )

@nb.njit(cache=True, fastmath=True)
def _extreme_indices_dirs(points: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    m = dirs.shape[0]
    out = np.empty(m, dtype=np.int64)

    for j in range(m):
        dx = dirs[j, 0]
        dy = dirs[j, 1]
        best_idx = 0
        best_val = dx * points[0, 0] + dy * points[0, 1]
        for i in range(1, points.shape[0]):
            val = dx * points[i, 0] + dy * points[i, 1]
            if val > best_val:
                best_val = val
                best_idx = i
        out[j] = best_idx

    return out

class Solver:
    def __init__(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float64)
        self._dirs16 = np.ascontiguousarray(
            np.column_stack((np.cos(angles), np.sin(angles))), dtype=np.float64
        )

        dummy_points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
        )
        dummy_order = np.array([0, 2, 1, 3], dtype=np.int64)
        _monotonic_chain(dummy_points, dummy_order)
        _outside_inner_polygon_mask(dummy_points, dummy_points[:3])
        _extreme_indices_8(dummy_points)
        _extreme_indices_dirs(dummy_points, self._dirs16)

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        points = problem["points"]
        if (
            not isinstance(points, np.ndarray)
            or points.dtype != np.float64
            or not points.flags.c_contiguous
        ):
            points = np.ascontiguousarray(points, dtype=np.float64)
            problem["points"] = points

        n = points.shape[0]

        if n <= 128:
            order = np.lexsort((points[:, 1], points[:, 0]))
            vertices = _monotonic_chain(points, order)
            if vertices.shape[0] >= 3:
                return {"hull_vertices": vertices, "hull_points": points[vertices]}

        if n >= 64:
            ext = np.unique(_extreme_indices_8(points))

            if ext.shape[0] >= 3:
                ext_order = np.lexsort((points[ext, 1], points[ext, 0]))
                inner_vertices = _monotonic_chain(points, ext[ext_order])

                if inner_vertices.shape[0] >= 3:
                    keep = _outside_inner_polygon_mask(points, points[inner_vertices])
                    keep[ext] = True
                    candidate_idx = np.flatnonzero(keep)
                    candidate_count = candidate_idx.shape[0]

                    if candidate_count + 32 < n and candidate_count * 5 < n * 4:
                        reduced = points[candidate_idx]

                        if candidate_count >= 384:
                            ext2 = np.unique(_extreme_indices_dirs(reduced, self._dirs16))
                            if ext2.shape[0] >= 3:
                                ext2_order = np.lexsort(
                                    (reduced[ext2, 1], reduced[ext2, 0])
                                )
                                inner2 = _monotonic_chain(reduced, ext2[ext2_order])

                                if inner2.shape[0] >= 3:
                                    keep2 = _outside_inner_polygon_mask(
                                        reduced, reduced[inner2]
                                    )
                                    keep2[ext2] = True
                                    candidate_idx2 = np.flatnonzero(keep2)
                                    candidate_count2 = candidate_idx2.shape[0]

                                    if (
                                        candidate_count2 + 16 < candidate_count
                                        and candidate_count2 * 4 < candidate_count * 3
                                    ):
                                        candidate_idx = candidate_idx[candidate_idx2]
                                        candidate_count = candidate_count2
                                        reduced = points[candidate_idx]

                        if candidate_count <= 8192:
                            reduced_order = np.lexsort((reduced[:, 1], reduced[:, 0]))
                            vertices = candidate_idx[
                                _monotonic_chain(reduced, reduced_order)
                            ]
                            if vertices.shape[0] >= 3:
                                return {
                                    "hull_vertices": vertices,
                                    "hull_points": points[vertices],
                                }
                        else:
                            vertices = candidate_idx[ConvexHull(reduced).vertices]
                            return {
                                "hull_vertices": vertices,
                                "hull_points": points[vertices],
                            }

        vertices = ConvexHull(points).vertices
        return {"hull_vertices": vertices, "hull_points": points[vertices]}