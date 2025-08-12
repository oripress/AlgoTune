from typing import Any
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def water_filling_numba(alpha, P_total):
    n = len(alpha)
    
    # Alternative approach without full sorting
    # Find the water level directly
    w = 0.0
    
    # Start with an initial guess
    w = (P_total + np.sum(alpha)) / n
    
    # Iterative refinement (few iterations needed due to convexity)
    for _ in range(10):
        x = np.maximum(0.0, w - alpha)
        x_sum = np.sum(x)
        if x_sum > 1e-12:
            # Adjust water level based on constraint violation
            w += (P_total - x_sum) / np.sum(w > alpha)
        else:
            # All channels are below water level
            w = (P_total + np.sum(alpha)) / n
            break
    
    # Final computation
    x_opt = np.maximum(0.0, w - alpha)
    x_sum = np.sum(x_opt)
    
    if x_sum > 1e-9:
        x_opt *= P_total / x_sum
    
    return x_opt

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        alpha = np.asarray(problem["alpha"], dtype=float)
        P_total = float(problem["P_total"])
        n = alpha.size
        
        # Handle edge cases
        if n == 0 or P_total <= 0 or alpha.min() <= 0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        # Compute optimal allocation using Numba
        x_opt = water_filling_numba(alpha, P_total)
        
        # Compute capacity
        capacity = float(np.log(alpha + x_opt).sum())
        
        return {"x": x_opt.tolist(), "Capacity": capacity}