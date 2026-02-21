import numpy as np
from numba import njit

@njit
def get_sorted_indices(points):
    n = len(points)
    indices = np.argsort(points[:, 0])
    
    # Fix ties in X by sorting by Y
    i = 0
    while i < n:
        j = i + 1
        while j < n and points[indices[j], 0] == points[indices[i], 0]:
            j += 1
        if j - i > 1:
            # Insertion sort for the tied Y values
            for k in range(i + 1, j):
                key = indices[k]
                val = points[key, 1]
                l = k - 1
                while l >= i and points[indices[l], 1] > val:
                    indices[l + 1] = indices[l]
                    l -= 1
                indices[l + 1] = key
        i = j
    return indices.astype(np.int32)

@njit
def cross_product(o_x, o_y, a_x, a_y, b_x, b_y):
    return (a_x - o_x) * (b_y - o_y) - (a_y - o_y) * (b_x - o_x)

@njit
def compute_hull(points):
    n = len(points)
    if n < 3:
        return np.arange(n, dtype=np.int32)
        
    indices = get_sorted_indices(points)
    lower = np.empty(n, dtype=np.int32)
    lower_sz = 0
    for i in range(n):
        idx = indices[i]
        px, py = points[idx, 0], points[idx, 1]
        while lower_sz >= 2:
            idx1 = lower[lower_sz - 2]
            idx2 = lower[lower_sz - 1]
            if cross_product(points[idx1, 0], points[idx1, 1],
                             points[idx2, 0], points[idx2, 1],
                             px, py) <= 0:
                lower_sz -= 1
            else:
                break
        lower[lower_sz] = idx
        lower_sz += 1

    upper = np.empty(n, dtype=np.int32)
    upper_sz = 0
    for i in range(n - 1, -1, -1):
        idx = indices[i]
        px, py = points[idx, 0], points[idx, 1]
        while upper_sz >= 2:
            idx1 = upper[upper_sz - 2]
            idx2 = upper[upper_sz - 1]
            if cross_product(points[idx1, 0], points[idx1, 1],
                             points[idx2, 0], points[idx2, 1],
                             px, py) <= 0:
                upper_sz -= 1
            else:
                break
        upper[upper_sz] = idx
        upper_sz += 1

    hull_sz = lower_sz - 1 + upper_sz - 1
    hull = np.empty(hull_sz, dtype=np.int32)
    for i in range(lower_sz - 1):
        hull[i] = lower[i]
    for i in range(upper_sz - 1):
        hull[lower_sz - 1 + i] = upper[i]
        
    return hull

class Solver:
    def __init__(self):
        dummy_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        compute_hull(dummy_points)
        
    def solve(self, problem, **kwargs):
        points = problem["points"]
        hull_indices = compute_hull(points)
        hull_points = points[hull_indices]
        return {
            "hull_vertices": hull_indices.tolist(),
            "hull_points": hull_points.tolist()
        }