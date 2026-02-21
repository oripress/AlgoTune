import numpy as np
from numba import njit
from typing import Any

@njit(cache=True)
def find_theta(u):
    n = len(u)
    cssv = 0.0
    rho = 0
    for i in range(n):
        val = u[n - 1 - i]
        cssv += val
        if val > (cssv - 1.0) / (i + 1):
            rho = i
        else:
            cssv -= val
            break
    return (cssv - 1.0) / (rho + 1)

class Solver:
    def __init__(self):
        # Precompile the Numba function
        find_theta(np.array([1.0, 2.0], dtype=np.float64))

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        y = np.asarray(problem["y"])
        if y.ndim > 1:
            y = y.flatten()
            
        u = np.sort(y)
        theta = find_theta(u)
        
        x = y - theta
        np.maximum(x, 0.0, out=x)
        return {"solution": x}