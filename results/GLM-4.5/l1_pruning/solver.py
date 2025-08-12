import numpy as np
from typing import Any, Dict, List
from numba import njit

@njit(fastmath=True)
def _solve_l1_projection_numba(v, k):
    """Numba-optimized implementation"""
    n = len(v)
    
    # Take absolute values
    u = np.abs(v)
    
    # Find max value for binary search bounds
    max_val = 0.0
    for i in range(n):
        if u[i] > max_val:
            max_val = u[i]
    
    # Binary search for theta
    theta_low = 0.0
    theta_high = max_val
    
    for _ in range(50):  # 50 iterations for precision
        theta_mid = (theta_low + theta_high) * 0.5
        sum_threshold = 0.0
        
        # Compute sum
        for i in range(n):
            val = u[i] - theta_mid
            if val > 0:
                sum_threshold += val
        
        if sum_threshold > k:
            theta_low = theta_mid
        else:
            theta_high = theta_mid
    
    theta = (theta_low + theta_high) * 0.5
    
    # Apply soft thresholding
    w = np.zeros(n)
    for i in range(n):
        val = u[i] - theta
        if val > 0:
            if v[i] >= 0:
                w[i] = val
            else:
                w[i] = -val
    
    return w

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the L1 proximal operator problem: min_w ‖v-w‖² s.t. ‖w‖₁ ≤ k
        
        Highly optimized Numba implementation.
        
        :param problem: A dictionary with keys "v" (list of numbers) and "k" (hyperparameter)
        :return: A dictionary with key "solution" containing the optimal solution vector
        """
        v_list = problem["v"]
        k = float(problem["k"])
        
        # Handle edge case
        if k <= 0:
            return {"solution": [0.0] * len(v_list)}
        
        # Convert to numpy array
        v = np.array(v_list, dtype=np.float64)
        
        # Use numba-optimized function
        w = _solve_l1_projection_numba(v, k)
        
        return {"solution": w.tolist()}