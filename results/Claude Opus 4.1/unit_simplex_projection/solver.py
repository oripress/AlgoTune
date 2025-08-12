import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Highly optimized Euclidean projection onto the probability simplex.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        n = len(y)
        
        # Sort in descending order
        y_sorted = np.sort(y)[::-1]
        
        # Compute cumulative sum minus 1 directly
        cumsum_minus_1 = np.cumsum(y_sorted) - 1
        
        # Find rho using vectorized comparison
        # We want the largest i where y_sorted[i] > cumsum_minus_1[i] / (i+1)
        indices = np.arange(1, n + 1, dtype=np.float64)
        check = y_sorted > cumsum_minus_1 / indices
        
        # Find last True index using argmax on reversed array
        if check.any():
            rho = n - 1 - np.argmax(check[::-1])
            theta = cumsum_minus_1[rho] / (rho + 1)
        else:
            theta = cumsum_minus_1[-1] / n
        
        # Final projection
        result = np.clip(y - theta, 0, None)
        
        return {"solution": result}