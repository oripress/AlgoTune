import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Optimized Euclidean projection onto probability simplex.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        n = y.size
        
        # Sort in descending order (single operation)
        u = np.sort(y)
        u = u[::-1]
        
        # Compute cumulative sum - 1 and divide by index+1 in one go
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, n + 1, dtype=np.float64)
        cond = u - cssv / ind > 0
        rho = np.where(cond)[0][-1]
        
        # Compute threshold
        theta = cssv[rho] / (rho + 1.0)
        
        # Project
        x = y - theta
        x[x < 0] = 0
        
        return {"solution": x}