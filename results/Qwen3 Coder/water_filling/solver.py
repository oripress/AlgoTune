from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Solve the water-filling problem using analytic solution."""
        alpha = np.asarray(problem["alpha"], dtype=float)
        P_total = float(problem["P_total"])
        n = len(alpha)
        
        # Handle edge cases
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        # Sort indices by alpha values
        idx = np.argsort(alpha)
        a_sorted = alpha[idx]
        
        # Calculate prefix sums
        prefix = np.cumsum(a_sorted)
        
        # Find the water level
        w = None
        for k in range(1, n + 1):
            w_candidate = (P_total + prefix[k - 1]) / k
            if k == n or w_candidate <= a_sorted[k]:
                w = w_candidate
                break
        
        # Fallback if no water level found (should not happen)
        if w is None:
            w = (P_total + prefix[-1]) / n
        
        # Calculate optimal allocation
        x_opt = np.maximum(0.0, w - alpha)
        
        # Rescale to match exact budget
        current_sum = np.sum(x_opt)
        if current_sum > 0:
            scaling_factor = P_total / current_sum
            x_scaled = x_opt * scaling_factor
        else:
            x_scaled = x_opt
        
        # Ensure non-negativity
        x_final = np.maximum(x_scaled, 0.0)
        
        # Final rescale if needed
        final_sum = np.sum(x_final)
        if final_sum > 0 and not np.isclose(final_sum, P_total):
            x_final *= P_total / final_sum
        
        # Calculate capacity
        safe_x = np.maximum(x_final, 0)
        capacity_terms = np.log(alpha + safe_x)
        capacity = float(np.sum(capacity_terms)) if np.all(np.isfinite(capacity_terms)) else float("nan")
        
        return {"x": x_final.tolist(), "Capacity": capacity}