import numpy as np
from typing import Any
import numba

@numba.njit(cache=True)
def l0_prune_numba(v, k):
    """
    Numba-compiled function to solve L0 proximal operator.
    Keeps k largest magnitude elements, zeros out the rest.
    """
    n = len(v)
    if k == 0:
        return np.zeros(n)
    if k >= n:
        return v.copy()
    
    # Create array of absolute values and indices
    abs_vals = np.abs(v)
    indices = np.arange(n)
    
    # Partial sort to find k-th largest element
    # We only need to partition, not fully sort
    pivot_idx = n - k
    
    # Simple partition-based selection (quickselect variant)
    left = 0
    right = n - 1
    
    while left < right:
        # Partition around a pivot
        pivot_val = abs_vals[right]
        i = left
        for j in range(left, right):
            if abs_vals[j] <= pivot_val:
                abs_vals[i], abs_vals[j] = abs_vals[j], abs_vals[i]
                indices[i], indices[j] = indices[j], indices[i]
                i += 1
        abs_vals[i], abs_vals[right] = abs_vals[right], abs_vals[i]
        indices[i], indices[right] = indices[right], indices[i]
        
        if i == pivot_idx:
            break
        elif i < pivot_idx:
            left = i + 1
        else:
            right = i - 1
    
    # Create result with zeros
    result = np.zeros(n)
    # Fill in the k largest elements
    for i in range(pivot_idx, n):
        result[indices[i]] = v[indices[i]]
    
    return result

class Solver:
    def __init__(self):
        # Pre-compile the numba function with a dummy call
        dummy = np.array([1.0, 2.0, 3.0])
        _ = l0_prune_numba(dummy, 1)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the L0 proximal operator by keeping k largest magnitude elements.
        Uses numba-compiled quickselect for optimal performance.
        """
        v = np.asarray(problem["v"], dtype=np.float64)
        k = problem["k"]
        
        result = l0_prune_numba(v, k)
        return {"solution": result.tolist()}