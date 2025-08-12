import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the water-filling problem using the analytical solution.
        
        The water-filling algorithm finds the optimal water level w such that:
        x_i = max(0, w - alpha_i) and sum(x_i) = P_total
        """
        alpha = np.asarray(problem["alpha"], dtype=float)
        P_total = float(problem["P_total"])
        n = alpha.size
        
        # Handle edge cases
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        # Sort alpha values in ascending order
        idx = np.argsort(alpha)
        a_sorted = alpha[idx]
        prefix = np.cumsum(a_sorted)
        
        # Find the optimal water level
        w = None
        for k in range(1, n + 1):
            w_candidate = (P_total + prefix[k - 1]) / k
            if k == n or w_candidate <= a_sorted[k]:
                w = w_candidate
                break
        
        if w is None:  # Should not happen
            w = (P_total + prefix[-1]) / n
        
        # Calculate optimal allocation
        x_opt = np.maximum(0.0, w - alpha)
        
        # Rescale to exactly match the budget constraint
        x_sum = np.sum(x_opt)
        if x_sum > 1e-9:
            scaling = P_total / x_sum
            x_opt *= scaling
        
        # Calculate capacity
        capacity = float(np.sum(np.log(alpha + x_opt)))
        
        return {"x": x_opt.tolist(), "Capacity": capacity}