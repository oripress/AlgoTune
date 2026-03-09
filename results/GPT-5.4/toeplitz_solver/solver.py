from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz

try:
    from scipy.linalg._solve_toeplitz import levinson as _levinson
except Exception:
    _levinson = None

_DENSE_N = 12

class Solver:
    def __init__(self):
        self._cache = {}

    def _get_cache(self, n: int):
        cache = self._cache.get(n)
        if cache is None:
            c = np.empty(n, dtype=np.float64)
            r = np.empty(n, dtype=np.float64)
            b = np.empty(n, dtype=np.float64)

            if n <= _DENSE_N:
                dense_vals = np.empty(2 * n - 1, dtype=np.float64)
                dense_idx = (
                    np.arange(n)[None, :] - np.arange(n)[:, None] + (n - 1)
                )
            else:
                dense_vals = None
                dense_idx = None

            if _levinson is not None and n > _DENSE_N:
                lev_vals = np.empty(2 * n - 1, dtype=np.float64)
            else:
                lev_vals = None

            cache = (c, r, b, dense_vals, dense_idx, lev_vals)
            self._cache[n] = cache
        return cache

    def solve(self, problem, **kwargs):
        b = problem["b"]
        n = len(b)

        if n == 0:
            return ()
        c = problem["c"]
        if n == 1:
            return (b[0] / c[0],)

        r = problem["r"]

        if n == 2:
            a = c[0]
            d = c[1]
            e = r[1]
            det = a * a - d * e
            b0 = b[0]
            b1 = b[1]
            return ((a * b0 - e * b1) / det, (-d * b0 + a * b1) / det)

        if n == 3:
            a = c[0]
            d = c[1]
            g = c[2]
            e = r[1]
            f = r[2]
            b0 = b[0]
            b1 = b[1]
            b2 = b[2]

            det = a * a * a - 2.0 * a * d * e + e * e * g + f * d * d - a * f * g
            x0 = (
                b0 * (a * a - d * e)
                + b1 * (d * f - a * e)
                + b2 * (e * e - a * f)
            ) / det
            x1 = (
                b0 * (e * g - a * d)
                + b1 * (a * a - f * g)
                + b2 * (d * f - a * e)
            ) / det
            x2 = (
                b0 * (d * d - a * g)
                + b1 * (e * g - a * d)
                + b2 * (a * a - d * e)
            ) / det
            return (x0, x1, x2)

        c_arr, r_arr, b_arr, dense_vals, dense_idx, lev_vals = self._get_cache(n)
        c_arr[:] = c
        r_arr[:] = r
        b_arr[:] = b

        if n <= _DENSE_N:
            dense_vals[: n - 1] = c_arr[:0:-1]
            dense_vals[n - 1 :] = r_arr
            return np.linalg.solve(dense_vals[dense_idx], b_arr)

        if lev_vals is not None:
            lev_vals[: n - 1] = r_arr[:0:-1]
            lev_vals[n - 1 :] = c_arr
            return _levinson(lev_vals, b_arr)[0]

        return solve_toeplitz((c_arr, r_arr), b_arr, check_finite=False)