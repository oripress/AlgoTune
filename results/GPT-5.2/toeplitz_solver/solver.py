from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_toeplitz

try:
    # Faster for Hermitian positive-definite Toeplitz (when applicable).
    from scipy.linalg import solveh_toeplitz  # type: ignore
except Exception:  # pragma: no cover
    solveh_toeplitz = None  # type: ignore

class Solver:
    def solve(self, problem: dict[str, list[float]], **kwargs: Any) -> Any:
        c_list = problem["c"]
        b_list = problem["b"]
        n = len(b_list)

        if n == 0:
            return np.empty(0, dtype=np.float64)
        if n == 1:
            return np.array([b_list[0] / c_list[0]], dtype=np.float64)

        # Convert once; ensures float64 and contiguous arrays.
        c = np.asarray(c_list, dtype=np.float64)
        b = np.asarray(b_list, dtype=np.float64)

        r_list = problem["r"]

        # If symmetric/Hermitian, try the specialized solver (often faster).
        if solveh_toeplitz is not None and r_list is c_list:
            return solveh_toeplitz(c, b, check_finite=False)

        r = np.asarray(r_list, dtype=np.float64)

        if solveh_toeplitz is not None and np.array_equal(r, c):
            try:
                return solveh_toeplitz(c, b, check_finite=False)
            except Exception:
                pass

        return solve_toeplitz((c, r), b, check_finite=False)