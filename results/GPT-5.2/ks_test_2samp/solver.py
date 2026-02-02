from __future__ import annotations

from typing import Any, Dict

import math

import numpy as np
from numba import njit
from scipy.stats import distributions
from scipy.stats._stats_py import _attempt_exact_2kssamp

@njit(cache=True)
def _ks_d_sorted(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample KS D statistic for two-sided test.

    Inputs must be 1D sorted float64 arrays.
    Computes sup_x |F1(x) - F2(x)| with right-continuous ECDFs
    (equivalent to SciPy's searchsorted(..., side='right')).
    """
    n1 = x.size
    n2 = y.size
    i = 0
    j = 0
    maxdiff = 0.0
    mindiff = 0.0
    inv_n1 = 1.0 / n1
    inv_n2 = 1.0 / n2

    while i < n1 or j < n2:
        if j == n2:
            v = x[i]
        elif i == n1:
            v = y[j]
        else:
            xv = x[i]
            yv = y[j]
            v = xv if xv <= yv else yv

        while i < n1 and x[i] == v:
            i += 1
        while j < n2 and y[j] == v:
            j += 1

        diff = i * inv_n1 - j * inv_n2
        if diff > maxdiff:
            maxdiff = diff
        if diff < mindiff:
            mindiff = diff

    d1 = maxdiff
    d2 = -mindiff
    return d1 if d1 >= d2 else d2

@njit(cache=True)
def _compute_prob_outside_square_nb(n: int, h: int) -> float:
    """Numba version of SciPy's _compute_prob_outside_square(n, h)."""
    P = 0.0
    k = int(n // h)  # floor(n/h) for positive integers
    while k >= 0:
        p1 = 1.0
        kh = k * h
        for j in range(h):
            p1 = (n - kh - j) * p1 / (n + kh + j + 1.0)
        P = p1 * (1.0 - P)
        k -= 1
    return 2.0 * P

# Warm up compilation at import (init time not counted).
_ks_d_sorted(np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64))
_compute_prob_outside_square_nb(10, 1)

# Use internal _sf to avoid rv_continuous.sf argument parsing overhead.
_KSTWO_SF = distributions.kstwo._sf  # type: ignore[attr-defined]
_INT32_MAX = np.iinfo(np.int32).max
_MAX_AUTO_N = 10000

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        x = np.array(problem["sample1"], dtype=np.float64)
        y = np.array(problem["sample2"], dtype=np.float64)
        x.sort()
        y.sort()

        n1 = x.size
        n2 = y.size
        if n1 == 0 or n2 == 0:
            raise ValueError("Data passed to ks_2samp must not be empty")

        d = float(_ks_d_sorted(x, y))

        # method="auto" in SciPy: attempt exact if max(n1,n2) <= 10000 else asymp.
        use_exact = (n1 if n1 >= n2 else n2) <= _MAX_AUTO_N
        prob: float

        if use_exact:
            if n1 == n2:
                # Exact two-sided for equal sizes uses the fast "outside square" method.
                n = int(n1)
                # SciPy rounds d to multiples of 1/n using np.round(d*n).
                h = int(np.rint(d * n))  # np.rint matches np.round (ties-to-even)
                d = h / n
                if h == 0:
                    prob = 1.0
                else:
                    prob = float(_compute_prob_outside_square_nb(n, h))
                    # Fallback to asymp if numerical issues
                    if not (0.0 <= prob <= 1.0) or not np.isfinite(prob):
                        use_exact = False
            else:
                # General exact (rare for this benchmark; included for safety).
                g = math.gcd(int(n1), int(n2))
                n1g = n1 // g
                n2g = n2 // g
                if n1g >= _INT32_MAX / n2g:
                    use_exact = False
                else:
                    success, d2, prob2 = _attempt_exact_2kssamp(int(n1), int(n2), g, d, "two-sided")
                    d = float(d2)
                    if success:
                        prob = float(prob2)
                    else:
                        use_exact = False

        if not use_exact:
            m = float(n1) if n1 >= n2 else float(n2)
            n = float(n2) if n1 >= n2 else float(n1)
            en = m * n / (m + n)
            prob = float(_KSTWO_SF(d, np.round(en)))

        # Clip like SciPy
        if prob < 0.0:
            prob = 0.0
        elif prob > 1.0:
            prob = 1.0

        return {"statistic": d, "pvalue": prob}