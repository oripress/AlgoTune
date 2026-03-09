from __future__ import annotations

import cmath
from typing import Any

import numpy as np
import scipy.linalg as la

class Solver:
    @staticmethod
    def _sort_eigenvalues(vals: np.ndarray) -> list[complex]:
        n = vals.shape[0]
        if n == 1:
            return [complex(vals[0])]
        if n == 2:
            a = complex(vals[0])
            b = complex(vals[1])
            if a.real > b.real or (a.real == b.real and a.imag >= b.imag):
                return [a, b]
            return [b, a]
        order = np.lexsort((-vals.imag, -vals.real))
        return vals[order].tolist()

    def solve(self, problem, **kwargs) -> Any:
        A, B = problem
        if not (isinstance(A, np.ndarray) and A.dtype == np.float64):
            A = np.asarray(A, dtype=np.float64)
        if not (isinstance(B, np.ndarray) and B.dtype == np.float64):
            B = np.asarray(B, dtype=np.float64)

        n = A.shape[0]
        if n == 1:
            return [complex(A[0, 0] / B[0, 0])]

        if n == 2:
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            e, f = B[0, 0], B[0, 1]
            g, h = B[1, 0], B[1, 1]
            c2 = e * h - f * g
            if abs(c2) > 1e-14:
                c1 = -(a * h + d * e - b * g - c * f)
                c0 = a * d - b * c
                root = cmath.sqrt(c1 * c1 - 4.0 * c2 * c0)
                s = 0.5 / c2
                x = (-c1 + root) * s
                y = (-c1 - root) * s
                if x.real > y.real or (x.real == y.real and x.imag >= y.imag):
                    return [x, y]
                return [y, x]

        try:
            vals = np.linalg.eigvals(np.linalg.solve(B, A))
        except Exception:
            vals = la.eigvals(A, B, overwrite_a=False, check_finite=False)

        return self._sort_eigenvalues(vals)