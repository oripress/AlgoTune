from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

if njit is not None:

    @njit(cache=False)
    def _theta_first_break_sorted_asc(mu_asc: np.ndarray, k: float) -> float:
        """
        mu_asc: sorted ascending absolute values.

        Mirrors the reference loop which runs on descending mu:
            cumsum += mu[j]
            if mu[j] < (cumsum - k)/(j+1): theta = ...; break
        returning 0.0 if the condition never holds.
        """
        c = 0.0
        n = mu_asc.shape[0]
        for t in range(n):  # t=0 corresponds to largest element
            idx = n - 1 - t
            val = mu_asc[idx]
            c += val
            theta = (c - k) / (t + 1.0)
            if val < theta:  # strict inequality, as in the reference
                return theta
        return 0.0

else:  # pragma: no cover

    def _theta_first_break_sorted_asc(mu_asc: np.ndarray, k: float) -> float:
        mu_desc = mu_asc[::-1]
        cssv = np.cumsum(mu_desc)
        j = np.arange(1, mu_desc.size + 1, dtype=np.float64)
        theta_vec = (cssv - k) / j
        idx = np.flatnonzero(mu_desc < theta_vec)
        return float(theta_vec[int(idx[0])]) if idx.size else 0.0

class Solver:
    """
    Project v onto the L1 ball of radius k, matching the provided reference's
    threshold selection rule.

    Performance tricks:
      - return NumPy arrays (avoid .tolist)
      - avoid full sort when possible via partition + sorting only top-m
      - Numba-jit the threshold loop (compiled in __init__)
    """

    __slots__ = ()

    def __init__(self) -> None:
        if njit is not None:
            _theta_first_break_sorted_asc(np.array([1.0], dtype=np.float64), 0.5)

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        # Slightly faster than .get() in tight loops.
        v_in = problem["v"]
        k = float(problem["k"])

        v = np.asarray(v_in, dtype=np.float64).ravel()
        n = v.size
        if n == 0:
            return {"solution": np.empty(0, dtype=np.float64)}

        if k < 0.0:
            return {"solution": np.zeros(n, dtype=np.float64)}

        # Compute abs(v) into a reusable buffer.
        u = np.empty_like(v)
        np.abs(v, out=u)

        # Early exit (matches reference behavior).
        if u.sum() <= k:
            return {"solution": v}

        # Compute theta by mutating u (sort/partition) in-place, then recompute abs(v)
        # into u for the actual output (keeps output aligned with original order).
        if n <= 2048:
            u.sort()  # in-place ascending
            theta = float(_theta_first_break_sorted_asc(u, k))
        else:
            m = 64
            if m > n:
                m = n
            while True:
                kth = n - m
                u.partition(kth)
                top = u[kth:]  # contiguous slice of length m
                top.sort()  # sort only top-m ascending
                theta = float(_theta_first_break_sorted_asc(top, k))
                if theta != 0.0 or m == n:
                    break
                m2 = m << 1
                m = n if m2 > n else m2

        # Recompute abs(v) into u (since u was permuted during theta computation).
        np.abs(v, out=u)

        # Apply threshold & restore signs in-place.
        u -= theta
        np.maximum(u, 0.0, out=u)
        np.copysign(u, v, out=u)

        return {"solution": u}