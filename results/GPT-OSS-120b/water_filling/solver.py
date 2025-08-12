from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analytic water‑filling solution.

        Parameters
        ----------
        problem: dict
            {
                "alpha": list of positive floats,
                "P_total": positive float
            }

        Returns
        -------
        dict
            {
                "x": list of optimal power allocations,
                "Capacity": total capacity sum(log(alpha_i + x_i))
            }
        """
        # Extract and validate inputs
        alpha = np.asarray(problem.get("alpha", []), dtype=float)
        P_total = float(problem.get("P_total", 0.0))
        n = alpha.size

        # Invalid cases – return NaNs as reference does
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # ---------- Analytic water‑filling ----------
        # Sort alphas (only for determining water level)
        a_sorted = np.sort(alpha)               # sorted alphas
        prefix = np.cumsum(a_sorted)            # cumulative sums

        # Vectorized computation of candidate water levels for each k
        k_arr = np.arange(1, n + 1, dtype=float)          # 1 … n
        w_candidates = (P_total + prefix) / k_arr

        # Compare each candidate with the next larger alpha (or ∞ for the last)
        next_alpha = np.empty_like(a_sorted)
        next_alpha[:-1] = a_sorted[1:]
        next_alpha[-1] = np.inf
        cond = w_candidates <= next_alpha

        # First k where condition holds gives the correct water level
        k = np.argmax(cond) + 1                     # +1 because k starts at 1
        w = w_candidates[k - 1]

        # Compute raw allocation and enforce exact power budget
        x_opt = np.maximum(0.0, w - alpha)
        total_alloc = np.sum(x_opt)
        if total_alloc > 0:
            x_opt *= (P_total / total_alloc)
        else:
            x_opt = np.zeros_like(alpha)

        # Compute capacity
        capacity = float(np.sum(np.log(alpha + x_opt)))

        return {"x": x_opt.tolist(), "Capacity": capacity}
        # Compute raw allocation
        x_opt = np.maximum(0.0, w - alpha)

        # Numerical tweak: ensure exact power budget
        total_alloc = np.sum(x_opt)
        if total_alloc > 0:
            scaling = P_total / total_alloc
            x_opt *= scaling
        else:
            # Edge case: all allocations zero (should not happen for positive P_total)
            x_opt = np.zeros_like(alpha)

        # Compute capacity
        capacity = float(np.sum(np.log(alpha + x_opt)))

        return {"x": x_opt.tolist(), "Capacity": capacity}