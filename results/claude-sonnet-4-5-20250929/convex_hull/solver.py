import numpy as np
from typing import Any
import numba

@numba.njit(fastmath=True)
def _cross(ox, oy, ax, ay, bx, by):
    """Calculate the cross product of vectors OA and OB."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

@numba.njit(fastmath=True, cache=True, inline='always')
def _insertion_sort_indices(points, indices):
    """Sort indices lexicographically by points coordinates."""
    n = len(indices)
    for i in range(1, n):
        key = indices[i]
        key_x = points[key, 0]
        key_y = points[key, 1]
        j = i - 1
        
        while j >= 0:
            idx = indices[j]
            if points[idx, 0] > key_x or (points[idx, 0] == key_x and points[idx, 1] > key_y):
                indices[j + 1] = indices[j]
                j -= 1
            else:
                break
        
        indices[j + 1] = key

@numba.njit(fastmath=True, cache=True)
def _convex_hull_monotone_chain(points):
    """
    Compute convex hull using Andrew's monotone chain algorithm.
    Returns indices of hull vertices in counter-clockwise order.
    """
    n = points.shape[0]
    
    # Create index array and sort points lexicographically
    indices = np.arange(n, dtype=np.int32)
    _insertion_sort_indices(points, indices)
    
    # Build lower hull
    lower = np.empty(n, dtype=np.int32)
    lower_size = 0
    
    for i in range(n):
        idx = indices[i]
        px = points[idx, 0]
        py = points[idx, 1]
        
        while lower_size >= 2:
            idx1 = lower[lower_size - 2]
            idx2 = lower[lower_size - 1]
            # Inline cross product
            cross = (points[idx2, 0] - points[idx1, 0]) * (py - points[idx1, 1]) - \
                    (points[idx2, 1] - points[idx1, 1]) * (px - points[idx1, 0])
            if cross <= 0:
                lower_size -= 1
            else:
                break
        lower[lower_size] = idx
        lower_size += 1
    
    # Build upper hull
    upper = np.empty(n, dtype=np.int32)
    upper_size = 0
    
    for i in range(n - 1, -1, -1):
        idx = indices[i]
        px = points[idx, 0]
        py = points[idx, 1]
        
        while upper_size >= 2:
            idx1 = upper[upper_size - 2]
            idx2 = upper[upper_size - 1]
            # Inline cross product
            cross = (points[idx2, 0] - points[idx1, 0]) * (py - points[idx1, 1]) - \
                    (points[idx2, 1] - points[idx1, 1]) * (px - points[idx1, 0])
            if cross <= 0:
                upper_size -= 1
            else:
                break
        upper[upper_size] = idx
        upper_size += 1
    
    # Combine hulls (remove last point of each as they're duplicated)
    hull_size = lower_size - 1 + upper_size - 1
    hull_vertices = np.empty(hull_size, dtype=np.int32)
    
    for i in range(lower_size - 1):
        hull_vertices[i] = lower[i]
    
    for i in range(upper_size - 1):
        hull_vertices[lower_size - 1 + i] = upper[i]
    
    return hull_vertices

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Convex Hull problem using optimized Andrew's monotone chain algorithm.
        
        :param problem: A dictionary with keys 'n' and 'points'
        :return: A dictionary with keys 'hull_vertices' and 'hull_points'
        """
        points = np.array(problem["points"], dtype=np.float64)
        
        hull_vertices = _convex_hull_monotone_chain(points)
        hull_vertices_list = hull_vertices.tolist()
        hull_points = points[hull_vertices].tolist()
        
        return {
            "hull_vertices": hull_vertices_list,
            "hull_points": hull_points
        }