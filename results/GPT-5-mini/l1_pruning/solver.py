from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Project v onto the L1-ball of radius k:
            min_w ||v - w||_2^2  s.t. ||w||_1 <= k

        Uses the O(n log n) algorithm (Duchi et al.):
          - Work with absolute values u = |v|
          - If sum(u) <= k return v (already feasible)
          - Sort u in descending order and compute cumulative sums
          - Find rho = max{ j : s_j > (cumsum_j - k)/j }
          - theta = (cumsum_rho - k) / rho
          - w = sign(v) * max(u - theta, 0)
        """
        v_in = problem.get("v", [])
        k_in = problem.get("k", 0.0)

        v = np.asarray(v_in, dtype=np.float64).ravel()
        n = v.size
        if n == 0:
            return {"solution": []}

        # Parse k robustly
        try:
            k = float(k_in)
        except Exception:
            k = 0.0

        # Handle special k values
        if np.isnan(k) or (not np.isfinite(k) and k < 0):
            # treat NaN or -inf as zero-radius -> zero vector
            return {"solution": np.zeros_like(v).tolist()}
        if np.isinf(k) and k > 0:
            # infinite radius -> original vector is feasible
            return {"solution": v.tolist()}

        if k <= 0.0:
            return {"solution": np.zeros_like(v).tolist()}

        u = np.abs(v)
        sum_u = float(u.sum())

        # Already within L1 ball
        if sum_u <= k:
            return {"solution": v.tolist()}

        # Sort magnitudes descending and compute cumulative sums
        s = np.sort(u)[::-1]
        cumsum = np.cumsum(s)
        j = np.arange(1, n + 1, dtype=np.float64)

        # Candidate thetas
        t = (cumsum - k) / j

        # Find rho = last index where s > t
        mask = s > t
        if np.any(mask):
            rho = np.nonzero(mask)[0][-1]
            theta = float(t[rho])
        else:
            theta = float(t[-1])

        # Soft-threshold and restore signs
        w = np.sign(v) * np.maximum(u - theta, 0.0)

        return {"solution": w.tolist()}