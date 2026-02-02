from __future__ import annotations

from typing import Any

import math

import numpy as np

class Solver:
    """
    Fast solver for symmetric real eigenvalues.

    Main speed win vs reference: use np.linalg.eigvalsh (eigenvalues only).
    Additional wins: closed-form fast paths for tiny matrices (n<=3).
    """

    def __init__(self) -> None:
        self._eigvalsh = np.linalg.eigvalsh

    @staticmethod
    def _eigvals_3x3_sym(a: np.ndarray) -> list[float]:
        # Analytic eigenvalues for symmetric 3x3 (float64).
        a00 = float(a[0, 0])
        a01 = float(a[0, 1])
        a02 = float(a[0, 2])
        a11 = float(a[1, 1])
        a12 = float(a[1, 2])
        a22 = float(a[2, 2])

        p1 = a01 * a01 + a02 * a02 + a12 * a12
        if p1 == 0.0:
            # Diagonal matrix.
            x0, x1, x2 = a00, a11, a22
            # manual sort descending (3 elements)
            if x0 < x1:
                x0, x1 = x1, x0
            if x1 < x2:
                x1, x2 = x2, x1
            if x0 < x1:
                x0, x1 = x1, x0
            return [x0, x1, x2]

        q = (a00 + a11 + a22) * (1.0 / 3.0)
        a00m = a00 - q
        a11m = a11 - q
        a22m = a22 - q
        p2 = a00m * a00m + a11m * a11m + a22m * a22m + 2.0 * p1
        p = math.sqrt(p2 * (1.0 / 6.0))

        invp = 1.0 / p
        b00 = a00m * invp
        b01 = a01 * invp
        b02 = a02 * invp
        b11 = a11m * invp
        b12 = a12 * invp
        b22 = a22m * invp

        # det(B) for symmetric 3x3:
        detb = (
            b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b02 * b12)
            + b02 * (b01 * b12 - b02 * b11)
        )
        r = 0.5 * detb

        # Clamp to [-1, 1] for numerical stability.
        if r <= -1.0:
            phi = math.pi / 3.0
        elif r >= 1.0:
            phi = 0.0
        else:
            phi = math.acos(r) * (1.0 / 3.0)

        two_p = 2.0 * p
        eig1 = q + two_p * math.cos(phi)
        eig3 = q + two_p * math.cos(phi + (2.0 * math.pi / 3.0))
        eig2 = (3.0 * q) - eig1 - eig3

        # Ensure descending order (eig1 >= eig2 >= eig3 typically, but be safe)
        x0, x1, x2 = eig1, eig2, eig3
        if x0 < x1:
            x0, x1 = x1, x0
        if x1 < x2:
            x1, x2 = x2, x1
        if x0 < x1:
            x0, x1 = x1, x0
        return [x0, x1, x2]

    def solve(self, problem, **kwargs) -> Any:
        a = problem
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)

        n = a.shape[0]

        if n == 0:
            return []
        if n == 1:
            return [float(a[0, 0])]
        if n == 2:
            # For symmetric [[p, q], [q, r]]
            p = float(a[0, 0])
            q = float(a[0, 1])
            r = float(a[1, 1])
            tr = p + r
            s = math.sqrt((p - r) * (p - r) + 4.0 * q * q)
            l1 = 0.5 * (tr + s)
            l2 = 0.5 * (tr - s)
            return [l1, l2] if l1 >= l2 else [l2, l1]
        if n == 3:
            return self._eigvals_3x3_sym(a)

        w = self._eigvalsh(a)  # ascending
        out = w.tolist()
        out.reverse()
        return out