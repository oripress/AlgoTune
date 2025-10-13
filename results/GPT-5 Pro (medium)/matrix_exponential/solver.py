from __future__ import annotations

from typing import Any
import math

import numpy as np
from scipy.linalg import expm

class Solver:
    @staticmethod
    def _expm_2x2(A: np.ndarray) -> list[list[float]]:
        """
        Closed-form matrix exponential for 2x2 real matrices.

        Uses the identity:
        Let t = trace(A)/2, B = A - t I, and delta^2 = ((a-d)^2)/4 + b*c.
        Then:
          exp(A) = e^t [ cosh(delta) I + (sinh(delta)/delta) B ], if delta^2 >= 0
          exp(A) = e^t [ cos(delta) I + (sin(delta)/delta) B ],   if delta^2 < 0
        with the convention sinh(0)/0 = 1 and sin(0)/0 = 1.
        """
        a = float(A[0, 0])
        b = float(A[0, 1])
        c = float(A[1, 0])
        d = float(A[1, 1])

        t = 0.5 * (a + d)
        B00 = a - t
        B01 = b
        B10 = c
        B11 = d - t

        delta2 = ((a - d) * (a - d)) / 4.0 + b * c
        et = math.exp(t)

        if delta2 > 0.0:
            delta = math.sqrt(delta2)
            s_term = math.sinh(delta) / delta
            c_term = math.cosh(delta)
        elif delta2 < 0.0:
            mu = math.sqrt(-delta2)
            s_term = math.sin(mu) / mu
            c_term = math.cos(mu)
        else:
            s_term = 1.0
            c_term = 1.0

        E00 = et * (c_term + s_term * B00)
        E01 = et * (s_term * B01)
        E10 = et * (s_term * B10)
        E11 = et * (c_term + s_term * B11)

        return [[E00, E01], [E10, E11]]

    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the matrix exponential exp(A) for a given square matrix A.

        Fast paths:
        - 1x1: scalar exp
        - 2x2: closed-form formula
        General case:
        - SciPy's expm
        """
        A = np.asarray(problem["matrix"], dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input 'matrix' must be a square 2D array.")
        n = A.shape[0]

        if n == 1:
            val = math.exp(float(A[0, 0]))
            expA = [[val]]
        elif n == 2:
            expA = self._expm_2x2(A)
        else:
            expA_np = expm(A)
            if not np.all(np.isfinite(expA_np)):
                raise FloatingPointError("Non-finite values encountered in matrix exponential.")
            expA = expA_np.tolist()

        return {"exponential": expA}