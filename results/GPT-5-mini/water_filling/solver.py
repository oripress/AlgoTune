from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analytic water-filling solver.

        Solves:
            maximize sum(log(alpha_i + x_i))
            s.t. sum(x) = P_total, x >= 0

        Optimal form:
            x_i = max(0, w - alpha_i) where w (water level) is chosen so that sum(x) = P_total.
        Returns dict with keys "x" (list) and "Capacity" (float).
        """
        # Basic validation
        if not isinstance(problem, dict):
            return {"x": [], "Capacity": float("nan")}
        if "alpha" not in problem or "P_total" not in problem:
            return {"x": [], "Capacity": float("nan")}

        try:
            alpha = np.asarray(problem["alpha"], dtype=float)
            P_total = float(problem["P_total"])
        except Exception:
            # If conversion fails, return NaNs of appropriate length if possible
            try:
                n_bad = len(problem.get("alpha", []))
            except Exception:
                n_bad = 0
            return {"x": [float("nan")] * n_bad, "Capacity": float("nan")}

        n = alpha.size

        # Degenerate/invalid inputs -> return NaNs (consistent with reference)
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Sort alphas ascending and compute prefix sums to find water level w
        a_sorted = np.sort(alpha)
        prefix = np.cumsum(a_sorted)

        # Vectorized computation of water-level candidates:
        k_arr = np.arange(1, n + 1, dtype=float)
        w_candidates = (P_total + prefix) / k_arr  # candidate water levels for k=1..n

        if n == 1:
            w = float(w_candidates[0])
        else:
            # For k in 1..n-1 select first k where w_candidate <= a_sorted[k]
            # (a_sorted[k] is the next alpha in sorted order; arrays are 0-indexed)
            mask = w_candidates[:-1] <= a_sorted[1:]
            where = np.nonzero(mask)[0]
            if where.size > 0:
                w = float(w_candidates[where[0]])
            else:
                w = float(w_candidates[-1])
        # Compute optimal allocation
        x_opt = np.maximum(0.0, w - alpha)

        # Numerical tweak: scale to meet budget exactly (if non-zero)
        sum_x = float(np.sum(x_opt))
        if sum_x > 0.0:
            x_opt = x_opt * (P_total / sum_x)

        # Clip tiny negatives (safety) and final rescale if clipping changed sum
        x_opt = np.maximum(0.0, x_opt)
        sum_x = float(np.sum(x_opt))
        if sum_x > 0.0 and not np.isclose(sum_x, P_total, rtol=1e-12, atol=1e-12):
            x_opt = x_opt * (P_total / sum_x)

        # Compute capacity
        terms = np.log(alpha + x_opt)
        if not np.all(np.isfinite(terms)):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        capacity = float(np.sum(terms))

        return {"x": x_opt.tolist(), "Capacity": capacity}