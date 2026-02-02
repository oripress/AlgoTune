import numpy as np
from scipy.spatial import ConvexHull
from typing import Any
from numba import njit

@njit(cache=True, fastmath=True)
def filter_interior_octagon(points_x, points_y, n):
    """Filter out points inside an octagon formed by 8 extreme points."""
    if n <= 16:
        return np.arange(n, dtype=np.int64)
    
    # Find 8 extreme points (octagon vertices)
    min_x = points_x[0]
    max_x = points_x[0]
    min_y = points_y[0]
    max_y = points_y[0]
    min_sum = points_x[0] + points_y[0]
    max_sum = min_sum
    min_diff = points_x[0] - points_y[0]
    max_diff = min_diff
    
    min_x_idx = 0
    max_x_idx = 0
    min_y_idx = 0
    max_y_idx = 0
    min_sum_idx = 0
    max_sum_idx = 0
    min_diff_idx = 0
    max_diff_idx = 0
    
    for i in range(n):
        px, py = points_x[i], points_y[i]
        if px < min_x:
            min_x = px
            min_x_idx = i
        if px > max_x:
            max_x = px
            max_x_idx = i
        if py < min_y:
            min_y = py
            min_y_idx = i
        if py > max_y:
            max_y = py
            max_y_idx = i
        s = px + py
        d = px - py
        if s < min_sum:
            min_sum = s
            min_sum_idx = i
        if s > max_sum:
            max_sum = s
            max_sum_idx = i
        if d < min_diff:
            min_diff = d
            min_diff_idx = i
        if d > max_diff:
            max_diff = d
            max_diff_idx = i
    
    # Build octagon vertices in CCW order
    oct_x = np.array([points_x[min_y_idx], points_x[max_diff_idx], points_x[max_x_idx], 
                      points_x[max_sum_idx], points_x[max_y_idx], points_x[min_diff_idx],
                      points_x[min_x_idx], points_x[min_sum_idx]])
    oct_y = np.array([points_y[min_y_idx], points_y[max_diff_idx], points_y[max_x_idx], 
                      points_y[max_sum_idx], points_y[max_y_idx], points_y[min_diff_idx],
                      points_y[min_x_idx], points_y[min_sum_idx]])
    
    # Precompute edge vectors
    edge_dx = np.empty(8)
    edge_dy = np.empty(8)
    for j in range(8):
        j_next = (j + 1) % 8
        edge_dx[j] = oct_x[j_next] - oct_x[j]
        edge_dy[j] = oct_y[j_next] - oct_y[j]
    
    # Check each point - keep if outside octagon
    result = np.empty(n, dtype=np.int64)
    count = 0
    
    for i in range(n):
        px, py = points_x[i], points_y[i]
        inside = True
        for j in range(8):
            dpx = px - oct_x[j]
            dpy = py - oct_y[j]
            cross = edge_dx[j] * dpy - edge_dy[j] * dpx
            if cross < -1e-10:
                inside = False
                break
        if not inside:
            result[count] = i
            count += 1
    
    # Always include extreme points
    oct_indices = np.array([min_y_idx, max_diff_idx, max_x_idx, max_sum_idx,
                            max_y_idx, min_diff_idx, min_x_idx, min_sum_idx], dtype=np.int64)
    
    for idx in oct_indices:
        found = False
        for j in range(count):
            if result[j] == idx:
                found = True
                break
        if not found:
            result[count] = idx
            count += 1
    
    return result[:count]

class Solver:
    def __init__(self):
        # Warm up numba
        x = np.array([0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5, 0.25, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8])
        y = np.array([0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.25, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8])
        filter_interior_octagon(x, y, 17)
    
    def solve(self, problem, **kwargs) -> Any:
        points = problem["points"]
        
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=np.float64)
        
        n = len(points)
        
        if n < 3:
            return {"hull_vertices": list(range(n)), "hull_points": points.tolist()}
        
        if n > 20:
            points_x = np.ascontiguousarray(points[:, 0])
            points_y = np.ascontiguousarray(points[:, 1])
            
            candidate_indices = filter_interior_octagon(points_x, points_y, n)
            
            if len(candidate_indices) >= 3 and len(candidate_indices) < n:
                filtered_points = points[candidate_indices]
                hull = ConvexHull(filtered_points)
                hull_vertices = candidate_indices[hull.vertices].tolist()
                hull_points = points[hull_vertices].tolist()
                return {"hull_vertices": hull_vertices, "hull_points": hull_points}
        
        hull = ConvexHull(points)
        hull_vertices = hull.vertices.tolist()
        hull_points = points[hull.vertices].tolist()
        
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}