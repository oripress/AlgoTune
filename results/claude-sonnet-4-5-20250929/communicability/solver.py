import numpy as np
from scipy.linalg import expm
from numba import njit

@njit(cache=True)
def build_adjacency_matrix(adj_list_flat, offsets, n):
    """Build adjacency matrix using numba for speed."""
    A = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        start = offsets[u]
        end = offsets[u + 1]
        for idx in range(start, end):
            v = adj_list_flat[idx]
            A[u, v] = 1.0
    return A

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, dict[int, dict[int, float]]]:
        """
        Optimized communicability calculation using numba for matrix construction.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        # Flatten adjacency list for numba
        adj_list_flat = []
        offsets = [0]
        for neighbors in adj_list:
            adj_list_flat.extend(neighbors)
            offsets.append(len(adj_list_flat))
        
        adj_list_flat = np.array(adj_list_flat, dtype=np.int32)
        offsets = np.array(offsets, dtype=np.int32)
        
        # Build adjacency matrix with numba
        A = build_adjacency_matrix(adj_list_flat, offsets, n)
        
        # Compute matrix exponential
        exp_A = expm(A)
        
        # Fast dict conversion
        result = {}
        for u in range(n):
            result[u] = dict(enumerate(exp_A[u].tolist()))
        
        return {"communicability": result}