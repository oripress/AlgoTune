import numpy as np
from typing import Any
import numba

# This helper function implements the O(n) randomized algorithm from Duchi et al. (2008).
# It is JIT-compiled with Numba for performance.
# This version is optimized to perform summation during the partitioning pass
# and uses Numba-native random generation and final projection loops.
@numba.njit(fastmath=True, cache=True)
def _solve_simplex_projection_fast(y: np.ndarray) -> np.ndarray:
    """
    Finds the Euclidean projection of a point y onto the probability simplex.
    
    This function implements a randomized linear-time algorithm with an optimized
    partitioning scheme that computes the sum simultaneously.
    Average case complexity: O(n)
    """
    n = y.shape[0]
    y_p = y.copy()

    s = 0.0
    rho = 0
    
    start = 0
    end = n
    
    while start < end:
        # Use Numba-friendly np.random.randint. Upper bound is exclusive.
        pivot_idx = np.random.randint(start, end)
        pivot_val = y_p[pivot_idx]
        
        # Move pivot to the end for partitioning
        y_p[pivot_idx], y_p[end - 1] = y_p[end - 1], y_p[pivot_idx]
        
        # Partition y_p[start:end-1] around pivot_val.
        # Elements >= pivot_val are moved to the left.
        # Simultaneously, compute the sum of the elements that are moved.
        store_idx = start
        g_sum = 0.0
        for i in range(start, end - 1):
            val = y_p[i]
            if val >= pivot_val:
                g_sum += val
                y_p[i], y_p[store_idx] = y_p[store_idx], val
                store_idx += 1
        
        # Move pivot to its final sorted position
        y_p[end - 1], y_p[store_idx] = y_p[store_idx], y_p[end - 1]
        
        # The set G from the paper corresponds to y_p[start : store_idx + 1]
        delta_rho = store_idx - start + 1
        delta_s = g_sum + pivot_val
            
        # Check the condition from Duchi et al. (2008), Algorithm 3
        if (s + delta_s) - (rho + delta_rho) * pivot_val < 1.0:
            # Condition met: the elements in G are part of the support.
            s += delta_s
            rho += delta_rho
            start = store_idx + 1
        else:
            # Condition not met: the support is a subset of G.
            end = store_idx
            
    # Compute the threshold theta.
    theta = (s - 1.0) / rho
    
    # Apply the threshold using an explicit loop, which is faster in Numba.
    x = np.empty_like(y)
    for i in range(n):
        x[i] = max(y[i] - theta, 0.0)
        
    return x

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solves the Euclidean projection onto the probability simplex problem.
        """
        y = np.array(problem.get("y"), dtype=np.float64)
        
        # Call the JIT-compiled O(n) helper function.
        solution = _solve_simplex_projection_fast(y)
        
        return {"solution": solution}