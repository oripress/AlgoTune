import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        alpha = np.asarray(problem['alpha'], dtype=np.float64)
        P_total = float(problem['P_total'])
        n = alpha.size
        
        # Handle edge cases efficiently
        if n == 0 or P_total <= 0 or np.any(alpha <= 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        # Sort alpha in ascending order (faster than argsort + indexing)
        a_sorted = np.sort(alpha)
        prefix = np.cumsum(a_sorted)
        
        # Binary search for the breakpoint k
        low, high = 1, n
        while low < high:
            mid = (low + high) // 2
            # Check if we can include the mid-th channel
            if (P_total + prefix[mid-1]) > mid * a_sorted[mid]:
                low = mid + 1
            else:
                high = mid
        
        k = low
        w = (P_total + prefix[k-1]) / k
        
        # Compute allocations - analytical solution satisfies constraint exactly
        x_opt = np.maximum(w - alpha, 0.0)
        
        # Efficient capacity calculation without temporary arrays
        capacity = np.sum(np.log(alpha + x_opt))
        
        return {
            "x": x_opt.tolist(),
            "Capacity": float(capacity)
        }