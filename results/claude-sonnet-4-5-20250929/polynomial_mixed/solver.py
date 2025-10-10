import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def fast_sort_indices(real_parts, imag_parts):
    """Fast sorting using numba JIT compilation."""
    n = len(real_parts)
    indices = np.arange(n)
    
    # Bubble sort for small arrays, should be fast with JIT
    for i in range(n):
        for j in range(n - 1 - i):
            if (real_parts[indices[j]] < real_parts[indices[j+1]]) or \
               (real_parts[indices[j]] == real_parts[indices[j+1]] and 
                imag_parts[indices[j]] < imag_parts[indices[j+1]]):
                indices[j], indices[j+1] = indices[j+1], indices[j]
    
    return indices

class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        """
        Optimized polynomial root finder with JIT-compiled sorting.
        """
        coefficients = np.array(problem)
        computed_roots = np.roots(coefficients)
        
        # Use JIT-compiled sorting
        real_parts = computed_roots.real
        imag_parts = computed_roots.imag
        sort_indices = fast_sort_indices(real_parts, imag_parts)
        sorted_roots = computed_roots[sort_indices]
        
        return sorted_roots.tolist()