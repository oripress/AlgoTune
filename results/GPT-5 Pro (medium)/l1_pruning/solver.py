from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Project vector v onto the L1 ball of radius k:
            argmin_w ||v - w||_2^2 subject to ||w||_1 <= k

        Uses the O(n log n) algorithm by Duchi et al. (2008).
        Optimized to avoid unnecessary temporary arrays by using a binary search
        to find rho instead of forming full boolean masks.
        """
        v = np.asarray(problem.get("v"), dtype=float).ravel()
        k = float(problem.get("k"))

        # Handle edge cases
        if k <= 0 or v.size == 0:
            return {"solution": np.zeros_like(v).tolist()}

        u = np.abs(v)
        u_sum = u.sum()
        if u_sum <= k or not np.isfinite(k):
            return {"solution": v.tolist()}

        # Sort in descending order
        s = np.sort(u)[::-1]
        cssv = np.cumsum(s)

        # Binary search for rho: largest j such that s[j] > (cssv[j] - k) / (j + 1)
        lo = 0
        hi = s.size - 1
        rho = -1  # will end as hi after loop
        while lo <= hi:
            mid = (lo + hi) // 2
            # Check condition at mid (0-based index)
            if s[mid] > (cssv[mid] - k) / (mid + 1):
                rho = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if rho < 0:
            # Should not happen if u_sum > k, but guard anyway
            return {"solution": np.zeros_like(v).tolist()}

        theta = (cssv[rho] - k) / (rho + 1)

        # Soft-threshold in-place to reduce allocations
        sgn = np.sign(v)
        u -= theta
        np.maximum(u, 0.0, out=u)
        u *= sgn

        return {"solution": u.tolist()}