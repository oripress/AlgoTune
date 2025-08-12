import numpy as np
from typing import Any
from numba import njit

@njit(fastmath=True)
def solve_l0_pruning_numba(v, k):
    """Numba-optimized L0 pruning implementation with direct indexing."""
    n = len(v)
    
    if k <= 0:
        return np.zeros(n, dtype=np.float64)
    if k >= n:
        return v.copy()
    
    # Compute absolute values
    abs_v = np.abs(v)
    
    # Get indices of k largest elements directly
    indices = np.argpartition(abs_v, n - k)[n - k:]
    
    # Create result and set values directly
    result = np.zeros(n, dtype=np.float64)
    for idx in indices:
        result[idx] = v[idx]
    
    return result

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the L0 proximal operator problem: min_w ‖v-w‖² s.t. ‖w‖₀ ≤ k
        
        This implementation uses numba with direct indexing for maximum performance.
        
        :param problem: A dictionary of the problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the l0_pruning task.
        """
        # Extract inputs directly as contiguous array
        v = np.ascontiguousarray(np.asarray(problem["v"], dtype=np.float64))
        k = int(problem["k"])
        
        # Use numba-optimized function
        result = solve_l0_pruning_numba(v, k)
        
        return {"solution": result.tolist()}