import numpy as np
from numba import jit
from typing import Any

@jit(nopython=True)
def solve_small_array(v, k):
    n = len(v)
    # Create array of indices and sort by absolute values (descending)
    indices = np.arange(n)
    # Sort indices by absolute values in descending order
    for i in range(n):
        for j in range(i+1, n):
            if abs(v[indices[i]]) < abs(v[indices[j]]):
                indices[i], indices[j] = indices[j], indices[i]
    
    solution = np.zeros(n)
    for i in range(min(k, n)):
        idx = indices[i]
        solution[idx] = v[idx]
    return solution

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the L0 proximal operator problem: min_w ‖v-w‖² s.t. ‖w‖₀ ≤ k
        """
        v = problem["v"]
        k = problem["k"]
        n = len(v)
        
        # Handle edge cases
        if k <= 0:
            return {"solution": [0.0] * n}
        if k >= n:
            return {"solution": list(v)}
        
        # For small arrays, use numba JIT
        if n < 100:
            v_array = np.array(v, dtype=np.float64)
            solution = solve_small_array(v_array, k)
            return {"solution": solution.tolist()}
        
        # For larger arrays, use numpy with argpartition
        v_array = np.asarray(v, dtype=np.float64)
        solution = np.zeros(n, dtype=np.float64)
        # Use argpartition for O(n) complexity instead of full sort
        keep_indices = np.argpartition(np.abs(v_array), -k)[-k:]
        solution[keep_indices] = v_array[keep_indices]
        
        return {"solution": solution.tolist()}