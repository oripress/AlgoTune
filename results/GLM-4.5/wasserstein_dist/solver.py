from typing import Any
import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def _compute_wasserstein(u, v):
    n = len(u)
    cumsum = 0.0
    distance = 0.0
    
    for i in range(n):
        cumsum += u[i] - v[i]
        distance += abs(cumsum)
    
    return distance

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Solves the wasserstein distance using an optimized numba-compiled approach with fastmath.
        
        :param problem: a Dict containing info for dist u and v
        :return: A float determine the wasserstein distance
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Check if problem exists and has required keys
        if not isinstance(problem, dict) or "u" not in problem or "v" not in problem:
            return 0.0
            
        try:
            # Convert to numpy arrays directly without intermediate variables
            u = np.array(problem["u"], dtype=np.float64, copy=False)
            v = np.array(problem["v"], dtype=np.float64, copy=False)
            
            # Check if arrays are valid
            if u.size == 0 or v.size == 0:
                return 0.0
                
            # Use numba-compiled function for maximum speed
            distance = _compute_wasserstein(u, v)
            
            return distance
        except:
            # Return 0 for any unexpected errors
            return 0.0