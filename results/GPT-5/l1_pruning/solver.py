from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Project v onto the L1 ball of radius k:
            argmin_w ||v - w||_2^2  subject to ||w||_1 <= k

        Uses a fast iterative thresholding method (expected O(n) with few passes),
        with a guaranteed O(n log n) sorting-based fallback for worst cases.
        """
        v_list = problem.get("v", [])
        # Handle empty input gracefully
        if v_list is None:
            return {"solution": []}
        v = np.asarray(v_list, dtype=float).ravel()

        # Parse and sanitize k
        k_raw = problem.get("k", 0.0)
        try:
            k = float(k_raw)
        except Exception:
            k = 0.0
        if not np.isfinite(k):
            k = 0.0
        if k <= 0.0:
            # Projection onto L1 ball of radius <= 0 is the zero vector
            return {"solution": np.zeros_like(v).tolist()}

        # If already within the L1 ball, return v
        abs_v = np.abs(v)
        sum_abs = float(abs_v.sum())
        if sum_abs <= k:
            return {"solution": v.tolist()}

        n = abs_v.size

        # Small-n fast path: exact sort-based projection
        if n <= 64:
            u = abs_v.copy()
            u.sort()          # ascending in-place
            u = u[::-1]       # descending view
            cssv = np.cumsum(u)
            inv_j = 1.0 / np.arange(1.0, u.size + 1.0)
            theta_seq = (cssv - k) * inv_j
            rho = int(np.count_nonzero(u > theta_seq)) - 1
            if rho < 0:
                rho = u.size - 1
            tau = float(theta_seq[rho])
            d = np.empty_like(abs_v)
            np.subtract(abs_v, tau, out=d)
            np.maximum(d, 0.0, out=d)
            np.copysign(d, v, out=d)
            return {"solution": d.tolist()}

        # Iterative thresholding method
        # Initialize tau as if all entries are active
        tau = (sum_abs - k) / n
        if tau < 0.0:
            tau = 0.0
        d = np.empty_like(abs_v)
        prev_support = -1
        converged = False
        tol_k = 1e-9 * (1.0 + abs(k))
        # Cap iterations to avoid pathological cases; fallback to sort if needed
        for _ in range(32):
            # Compute positive parts and their sum
            np.subtract(abs_v, tau, out=d)
            np.maximum(d, 0.0, out=d)
            s_diff = float(d.sum())

            # If already close enough to target L1 sum, accept
            if abs(s_diff - k) <= tol_k:
                converged = True
                break

            # Compute current support only if needed
            support = int(np.count_nonzero(d))
            if support == 0:
                # All are below threshold (shouldn't happen since sum_abs > k)
                break

            # Fixed-point/Newton update for tau: F(tau) = sum(max(abs_v - tau, 0)) - k
            # F'(tau) = -support
            new_tau = tau + (s_diff - k) / support

            # Convergence checks: support stable and tau change tiny
            if support == prev_support and abs(new_tau - tau) <= 1e-12 * (1.0 + abs(tau)):
                tau = new_tau
                converged = True
                break

            prev_support = support
            tau = new_tau

        if not converged:
            # Check if current tau already tight enough
            np.subtract(abs_v, tau, out=d)
            np.maximum(d, 0.0, out=d)
            s_diff = float(d.sum())
            if not np.isfinite(tau) or abs(s_diff - k) > tol_k:
                # Fallback to sorting-based exact method (O(n log n))
                u = np.sort(abs_v)[::-1]  # descending
                cssv = np.cumsum(u)
                j = np.arange(1, u.size + 1)  # 1..n as int
                theta_seq = (cssv - k) / j
                # number of indices where u > theta
                rho = int(np.count_nonzero(u > theta_seq)) - 1
                if rho < 0:
                    rho = u.size - 1
                tau = float(theta_seq[rho])

                # Recompute positive parts with exact tau
                np.subtract(abs_v, tau, out=d)
                np.maximum(d, 0.0, out=d)

        # Apply signs in-place using copysign
        np.copysign(d, v, out=d)

        return {"solution": d.tolist()}