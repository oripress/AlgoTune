import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve the water-filling problem using the analytic solution."""
        alpha = np.asarray(problem["alpha"], dtype=float)
        P_total = float(problem["P_total"])
        n = len(alpha)
        
        # Handle edge cases
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        # Sort alpha values in ascending order
        idx = np.argsort(alpha)
        a_sorted = alpha[idx]
        prefix = np.cumsum(a_sorted)
        
        # Find the water level w
        w = None
        for k in range(1, n + 1):
            w_candidate = (P_total + prefix[k - 1]) / k
            if k == n or w_candidate <= a_sorted[k]:
                w = w_candidate
                break
        
        if w is None:  # should not happen
            w = (P_total + prefix[-1]) / n
        
        # Compute optimal allocation
        x_opt = np.maximum(0.0, w - alpha)
        
        # Rescale to match exact budget
        current_sum = np.sum(x_opt)
        if current_sum > 0:
            x_opt = x_opt * (P_total / current_sum)
        
        # Compute capacity
        capacity = float(np.sum(np.log(alpha + x_opt)))
        
        return {"x": x_opt.tolist(), "Capacity": capacity}