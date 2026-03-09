from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

try:
    from scipy.linalg.lapack import dpotrf
except Exception:  # pragma: no cover
    dpotrf = None

@njit(cache=True, fastmath=True)
def _chol_numba(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                v = A[i, i] - s
                if v < 0.0 and v > -1e-12:
                    v = 0.0
                L[i, j] = math.sqrt(v)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

class Solver:
    def __init__(self) -> None:
        try:
            _chol_numba(np.eye(2, dtype=np.float64))
        except Exception:
            pass

    def solve(self, problem, **kwargs) -> Any:
        A = problem["matrix"]
        n = A.shape[0]
        sqrt = math.sqrt

        if n == 1:
            L = np.empty((1, 1), dtype=np.float64)
            L[0, 0] = sqrt(float(A[0, 0]))
            return {"Cholesky": {"L": L}}

        if n == 2:
            a00 = float(A[0, 0])
            a10 = float(A[1, 0])
            a11 = float(A[1, 1])

            l00 = sqrt(a00)
            l10 = a10 / l00
            l11 = sqrt(a11 - l10 * l10)

            L = np.zeros((2, 2), dtype=np.float64)
            L[0, 0] = l00
            L[1, 0] = l10
            L[1, 1] = l11
            return {"Cholesky": {"L": L}}

        if n == 3:
            a00 = float(A[0, 0])
            a10 = float(A[1, 0])
            a11 = float(A[1, 1])
            a20 = float(A[2, 0])
            a21 = float(A[2, 1])
            a22 = float(A[2, 2])

            l00 = sqrt(a00)
            l10 = a10 / l00
            l20 = a20 / l00
            l11 = sqrt(a11 - l10 * l10)
            l21 = (a21 - l20 * l10) / l11
            l22 = sqrt(a22 - l20 * l20 - l21 * l21)

            L = np.zeros((3, 3), dtype=np.float64)
            L[0, 0] = l00
            L[1, 0] = l10
            L[1, 1] = l11
            L[2, 0] = l20
            L[2, 1] = l21
            L[2, 2] = l22
            return {"Cholesky": {"L": L}}

        if n <= 32:
            L = _chol_numba(A)
            return {"Cholesky": {"L": L}}

        if dpotrf is not None and A.dtype == np.float64:
            L, info = dpotrf(A, lower=1, overwrite_a=0, clean=1)
            if info == 0:
                return {"Cholesky": {"L": L}}

        L = np.linalg.cholesky(A)
        return {"Cholesky": {"L": L}}