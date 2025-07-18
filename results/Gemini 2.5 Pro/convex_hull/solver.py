from typing import Any
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module
import numba

@numba.njit(cache=True)
def _find_extreme_indices_numba(points: np.ndarray) -> np.ndarray:
    """
    Finds the indices of the 8 extreme points (min/max of x, y, x+y, x-y)
    in a single pass over the data, avoiding large temporary arrays.
    """
    # Initialize values and indices with the first point
    min_x_idx, max_x_idx = 0, 0
    min_y_idx, max_y_idx = 0, 0
    min_s_idx, max_s_idx = 0, 0
    min_d_idx, max_d_idx = 0, 0

    p0 = points[0]
    min_x, max_x = p0[0], p0[0]
    min_y, max_y = p0[1], p0[1]
    min_s, max_s = p0[0] + p0[1], p0[0] + p0[1]
    min_d, max_d = p0[0] - p0[1], p0[0] - p0[1]

    for i in range(1, len(points)):
        p = points[i]
        x, y = p[0], p[1]
        
        if x < min_x: min_x, min_x_idx = x, i
        elif x > max_x: max_x, max_x_idx = x, i
        
        if y < min_y: min_y, min_y_idx = y, i
        elif y > max_y: max_y, max_y_idx = y, i
        
        s = x + y
        if s < min_s: min_s, min_s_idx = s, i
        elif s > max_s: max_s, max_s_idx = s, i
        
        d = x - y
        if d < min_d: min_d, min_d_idx = d, i
        elif d > max_d: max_d, max_d_idx = d, i
    
    return np.array([min_x_idx, max_x_idx, min_y_idx, max_y_idx, min_s_idx, max_s_idx, min_d_idx, max_d_idx], dtype=np.int64)

@numba.njit(fastmath=True, cache=True)
def _filter_candidates_numba(points: np.ndarray, filter_polygon: np.ndarray) -> np.ndarray:
    """
    Fast filtering of points by checking if they lie outside a convex filter_polygon.
    """
    num_points = points.shape[0]
    num_verts = filter_polygon.shape[0]
    candidate_indices_buffer = np.empty(num_points, dtype=np.int64)
    candidate_count = 0

    for i in range(num_points):
        p = points[i]
        is_strictly_inside = True
        for j in range(num_verts):
            p1 = filter_polygon[j]
            p2 = filter_polygon[(j + 1) % num_verts]
            cross_product = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross_product <= 1e-12:
                is_strictly_inside = False
                break
        
        if not is_strictly_inside:
            candidate_indices_buffer[candidate_count] = i
            candidate_count += 1
            
    return candidate_indices_buffer[:candidate_count]

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        points = np.array(problem["points"], dtype=np.float64)
        n = len(points)

        if n <= 16:
            if n < 3:
                return {"hull_vertices": list(range(n)), "hull_points": points.tolist()}
            hull = ConvexHull(points)
            return {"hull_vertices": hull.vertices.tolist(), "hull_points": points[hull.vertices].tolist()}

        # 1. Find extreme points in 8 directions using a single Numba pass.
        extreme_indices_list = _find_extreme_indices_numba(points)
        
        # 2. Create the filter polygon from the unique extreme points.
        extreme_indices = np.unique(extreme_indices_list)
        extreme_points = points[extreme_indices]

        # 3. Sort the extreme points counter-clockwise.
        centroid = np.mean(extreme_points, axis=0)
        angles = np.arctan2(extreme_points[:, 1] - centroid[1], extreme_points[:, 0] - centroid[0])
        filter_polygon = extreme_points[np.argsort(angles)]

        # 4. Filter points using the highly optimized Numba function.
        candidate_indices = _filter_candidates_numba(points, filter_polygon)
        
        if len(candidate_indices) < 3:
            hull = ConvexHull(points)
            final_indices = hull.vertices
        else:
            candidate_points = points[candidate_indices]
            # 5. Run ConvexHull on the much smaller set of candidate points.
            hull_of_candidates = ConvexHull(candidate_points)
            # 6. Map local indices back to original indices.
            final_indices = candidate_indices[hull_of_candidates.vertices]

        return {
            "hull_vertices": final_indices.tolist(),
            "hull_points": points[final_indices].tolist()
        }