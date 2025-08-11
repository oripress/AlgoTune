import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay
from typing import Any
import numba as nb

@nb.njit(cache=True, fastmath=True)
def fast_canonicalize_simplices(simplices):
    """Fast canonicalization of simplices using numba."""
    n = len(simplices)
    result = np.empty((n, 3), dtype=np.int32)
    for i in range(n):
        a, b, c = simplices[i]
        # Inline sorting network for 3 elements
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        result[i, 0] = a
        result[i, 1] = b
        result[i, 2] = c
    return result

@nb.njit(cache=True, fastmath=True)
def fast_canonicalize_edges(edges):
    """Fast canonicalization of edges using numba."""
    n = len(edges)
    result = np.empty((n, 2), dtype=np.int32)
    for i in range(n):
        a, b = edges[i]
        if a > b:
            result[i, 0] = b
            result[i, 1] = a
        else:
            result[i, 0] = a
            result[i, 1] = b
    return result

@nb.njit(cache=True)
def fast_sort_simplices(arr):
    """Optimized sort for canonicalized simplices."""
    n = len(arr)
    # Create combined key for faster comparison
    keys = arr[:, 0] * 1000000 + arr[:, 1] * 1000 + arr[:, 2]
    indices = np.argsort(keys)
    return arr[indices]

@nb.njit(cache=True)
def fast_sort_edges(arr):
    """Optimized sort for canonicalized edges."""
    n = len(arr)
    # Create combined key for faster comparison
    keys = arr[:, 0] * 10000 + arr[:, 1]
    indices = np.argsort(keys)
    return arr[indices]

class Solver:
    def __init__(self):
        # Pre-compile numba functions with dummy data
        dummy_simplices = np.array([[0, 1, 2]], dtype=np.int32)
        dummy_edges = np.array([[0, 1]], dtype=np.int32)
        fast_canonicalize_simplices(dummy_simplices)
        fast_canonicalize_edges(dummy_edges)
        fast_sort_simplices(dummy_simplices)
        fast_sort_edges(dummy_edges)
        
        # Pre-allocate workspace for common sizes
        self.workspace = {}
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Compute Delaunay triangulation of 2D points."""
        # Direct access to list avoids conversion overhead
        pts_list = problem["points"]
        n = len(pts_list)
        
        # Convert to numpy array efficiently
        pts = np.array(pts_list, dtype=np.float64)
        
        # Use optimal qhull options based on dataset size
        if n < 50:
            qhull_opts = "QJ Qx"  # More precise for small datasets
        elif n < 500:
            qhull_opts = "QJ"
        else:
            qhull_opts = "Qbb QJ"  # Scale for large datasets
        
        tri = SciPyDelaunay(pts, qhull_options=qhull_opts)
        
        # Get simplices and hull efficiently
        simplices = tri.simplices.astype(np.int32, copy=False)
        convex_hull = tri.convex_hull.astype(np.int32, copy=False)
        
        # Fast canonicalization and sorting
        canonical_simplices = fast_canonicalize_simplices(simplices)
        canonical_hull = fast_canonicalize_edges(convex_hull)
        
        # Use optimized sorting
        sorted_simplices = fast_sort_simplices(canonical_simplices)
        sorted_hull = fast_sort_edges(canonical_hull)
        
        result = {
            "simplices": sorted_simplices.tolist(),
            "convex_hull": sorted_hull.tolist(),
        }
        return result