import numpy as np
from typing import Any
from numba import njit

@njit(cache=True)
def subproblem_sol_numba(vn, z):
    """Numba-compiled version of subproblem_sol for speed."""
    mu = np.sort(vn)[::-1]
    cumsum = 0.0
    theta = 0.0
    n = len(mu)
    
    for j in range(n):
        cumsum += mu[j]
        if mu[j] < (cumsum - z) / (j + 1):
            theta = (cumsum - z) / (j + 1)
            break
    
    # Apply thresholding
    w = np.maximum(vn - theta, 0.0)
    return w

class Solver:
    def __init__(self):
        """Pre-compile numba functions."""
        # Trigger compilation with a dummy call
        dummy = np.array([1.0, 2.0], dtype=np.float64)
        subproblem_sol_numba(dummy, 1.0)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the L1 proximal operator problem efficiently.
        
        :param problem: A dictionary with keys "v" (vector) and "k" (constraint)
        :return: A dictionary with key "solution"
        """
        v = np.array(problem.get("v"), dtype=np.float64)
        k = problem.get("k")
        
        # Work with absolute values
        u = np.abs(v)
        
        # Solve the subproblem
        b = subproblem_sol_numba(u, k)
        
        # Restore signs
        pruned = b * np.sign(v)
        
        return {"solution": pruned.tolist()}