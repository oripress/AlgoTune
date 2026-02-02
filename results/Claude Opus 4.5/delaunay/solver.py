import numpy as np
from scipy.spatial import Delaunay
from numba import njit

@njit(cache=True)
def canonicalize_all(simplices, hull):
    """Canonicalize simplices and hull in one function to reduce overhead."""
    n_simp = simplices.shape[0]
    n_hull = hull.shape[0]
    
    # Sort each simplex row (3 elements)
    sorted_simp = np.empty((n_simp, 3), dtype=simplices.dtype)
    for i in range(n_simp):
        a, b, c = simplices[i, 0], simplices[i, 1], simplices[i, 2]
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        sorted_simp[i, 0] = a
        sorted_simp[i, 1] = b
        sorted_simp[i, 2] = c
    
    # Sort each hull edge row (2 elements)
    sorted_hull = np.empty((n_hull, 2), dtype=hull.dtype)
    for i in range(n_hull):
        a, b = hull[i, 0], hull[i, 1]
        if a > b:
            a, b = b, a
        sorted_hull[i, 0] = a
        sorted_hull[i, 1] = b
    
    # Lexsort simplices
    if n_simp > 0:
        M_simp = np.int64(np.max(sorted_simp) + 1)
        keys_simp = sorted_simp[:, 0] * M_simp * M_simp + sorted_simp[:, 1] * M_simp + sorted_simp[:, 2]
        indices_simp = np.argsort(keys_simp)
        sorted_simp = sorted_simp[indices_simp]
    
    # Lexsort hull
    if n_hull > 0:
        M_hull = np.int64(np.max(sorted_hull) + 1)
        keys_hull = sorted_hull[:, 0] * M_hull + sorted_hull[:, 1]
        indices_hull = np.argsort(keys_hull)
        sorted_hull = sorted_hull[indices_hull]
    
    return sorted_simp, sorted_hull

class Solver:
    def solve(self, problem, **kwargs):
        pts = np.asarray(problem["points"], dtype=np.float64)
        
        # Use qhull_options for potential speedup
        tri = Delaunay(pts, qhull_options="Qbb Qc Qt")
        
        simplices, hull = canonicalize_all(tri.simplices, tri.convex_hull)
        
        return {
            "simplices": simplices,
            "convex_hull": hull,
        }