import numpy as np
from numba import njit
from scipy.sparse.linalg import eigsh

@njit
def find_two_smallest_abs(eigenvalues):
    """Find indices of two smallest absolute values using Numba."""
    n = len(eigenvalues)
    abs_vals = np.abs(eigenvalues)
    
    # Find two smallest
    if abs_vals[0] <= abs_vals[1]:
        idx1, idx2 = 0, 1
        min1, min2 = abs_vals[0], abs_vals[1]
    else:
        idx1, idx2 = 1, 0
        min1, min2 = abs_vals[1], abs_vals[0]
    
    for i in range(2, n):
        val = abs_vals[i]
        if val < min1:
            min2 = min1
            idx2 = idx1
            min1 = val
            idx1 = i
        elif val < min2:
            min2 = val
            idx2 = i
    
    # Return the actual eigenvalues sorted by absolute value
    if abs_vals[idx1] <= abs_vals[idx2]:
        return [eigenvalues[idx1], eigenvalues[idx2]]
    else:
        return [eigenvalues[idx2], eigenvalues[idx1]]

class Solver:
    def __init__(self):
        # Warm up JIT compilation
        dummy = np.array([1.0, -0.5, 2.0, -0.1])
        find_two_smallest_abs(dummy)
    
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """Find the two eigenvalues closest to zero using optimized computation."""
        matrix = np.array(problem["matrix"], dtype=np.float64)
        n = matrix.shape[0]
        
        # For small matrices, use numpy with numba optimization
        if n <= 20:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return find_two_smallest_abs(eigenvalues)
        
        # For larger matrices, use targeted eigenvalue computation
        try:
            # Request 4 eigenvalues near 0 to ensure we get the 2 closest
            k = min(4, n-1)
            eigenvalues, _ = eigsh(matrix, k=k, sigma=0.0, which='LM')
            return find_two_smallest_abs(eigenvalues)
        except:
            # Fallback
            eigenvalues = np.linalg.eigvalsh(matrix)
            return find_two_smallest_abs(eigenvalues)