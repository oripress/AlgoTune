from __future__ import annotations

from typing import Any, Dict
import numpy as np

class Solver:
    @staticmethod
    def _water_filling_optimal(alpha: np.ndarray, P_total: float) -> np.ndarray:
        """
        Analytic water-filling solution:
            x_i* = max(0, w − α_i), where w is chosen so that Σ x_i* = P_total.
        """
        n = alpha.size
        if n == 0:
            return np.array([], dtype=float)

        # Sort alphas ascending
        a_sorted = np.sort(alpha)
        prefix = np.cumsum(a_sorted)

        # Compute candidate water levels for k = 1..n
        k_arr = np.arange(1, n + 1, dtype=float)
        w_candidates = (P_total + prefix) / k_arr

        # Build mask where w_k <= next alpha (with last always True)
        mask = np.empty(n, dtype=bool)
        if n > 1:
            mask[:-1] = w_candidates[:-1] <= a_sorted[1:]
        mask[-1] = True

        # Find the first k satisfying the condition
        k_idx = int(np.argmax(mask))
        w = w_candidates[k_idx]

        # Allocate
        x = np.maximum(0.0, w - alpha)

        # Rescale to exactly meet budget (numerical robustness)
        s = float(np.sum(x))
        if s > 0.0:
            x *= (P_total / s)
        return x

    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        alpha = np.asarray(problem.get("alpha", []), dtype=float)
        P_total = float(problem.get("P_total", 0.0))
        n = alpha.size

        # Handle invalid/degenerate inputs similarly to the reference
        if n == 0 or P_total <= 0.0 or not np.all(alpha > 0.0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Compute optimal allocation via analytic water-filling
        x = self._water_filling_optimal(alpha, P_total)

        # Capacity using natural logarithm
        capacity = float(np.sum(np.log(alpha + x)))

        return {"x": x.tolist(), "Capacity": capacity}