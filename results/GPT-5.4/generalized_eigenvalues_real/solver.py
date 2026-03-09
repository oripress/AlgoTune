from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        A, B = problem
        n = len(A)

        if n == 1:
            return [float(A[0][0] / B[0][0])]

        if n == 2:
            a11 = float(A[0][0])
            a12 = float(A[0][1])
            a22 = float(A[1][1])

            b11 = float(B[0][0])
            b12 = float(B[0][1])
            b22 = float(B[1][1])

            c2 = b11 * b22 - b12 * b12
            c1 = -(a11 * b22 + a22 * b11 - 2.0 * a12 * b12)
            c0 = a11 * a22 - a12 * a12

            disc = c1 * c1 - 4.0 * c2 * c0
            if disc < 0.0 and disc > -1e-15 * (c1 * c1 + abs(c2 * c0) + 1.0):
                disc = 0.0
            root = math.sqrt(disc)
            scale = 0.5 / c2
            l1 = (-c1 + root) * scale
            l2 = (-c1 - root) * scale
            if l1 >= l2:
                return [l1, l2]
            return [l2, l1]

        A_arr = np.asarray(A, dtype=np.float64)
        B_arr = np.asarray(B, dtype=np.float64)

        L = np.linalg.cholesky(B_arr)
        X = solve_triangular(L, A_arr, lower=True, check_finite=False)
        A_tilde = solve_triangular(L, X.T, lower=True, check_finite=False).T
        vals = np.linalg.eigvalsh(A_tilde)
        return vals[::-1].tolist()