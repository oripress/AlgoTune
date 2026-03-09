from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.linalg import eigh as scipy_eigh
    try:
        from scipy.linalg.lapack import dsygvd as scipy_dsygvd
    except Exception:  # pragma: no cover
        scipy_dsygvd = None
except Exception:  # pragma: no cover
    scipy_eigh = None
    scipy_dsygvd = None

class Solver:
    def __init__(self) -> None:
        self._asarray = np.asarray
        self._sqrt = np.sqrt
        self._cholesky = np.linalg.cholesky
        self._eigh = np.linalg.eigh
        self._solve = np.linalg.solve
        self._scipy_eigh = scipy_eigh
        self._scipy_dsygvd = scipy_dsygvd
        self._use_dsygvd = scipy_dsygvd is not None

    def _fallback(self, A: np.ndarray, B: np.ndarray) -> tuple[list[float], list[list[float]]]:
        L = self._cholesky(B)
        Y = self._solve(L, A)
        Atilde = self._solve(L, Y.T).T
        eigenvalues, eigenvectors = self._eigh(Atilde)
        eigenvectors = self._solve(L.T, eigenvectors)

        Bv = B @ eigenvectors
        norms = self._sqrt(np.sum(eigenvectors * Bv, axis=0))
        norms[norms == 0.0] = 1.0
        eigenvectors /= norms

        return (eigenvalues[::-1].tolist(), eigenvectors[:, ::-1].T.tolist())

    def _solve_2x2(self, A: np.ndarray, B: np.ndarray) -> tuple[list[float], list[list[float]]]:
        a = float(A[0, 0])
        b = float(A[0, 1])
        c = float(A[1, 1])

        d = float(B[0, 0])
        e = float(B[0, 1])
        f = float(B[1, 1])

        p = d * f - e * e
        q = 2.0 * b * e - a * f - c * d
        r = a * c - b * b
        disc = q * q - 4.0 * p * r
        if disc < 0.0 and disc > -1e-14 * (q * q + abs(4.0 * p * r) + 1.0):
            disc = 0.0
        if disc < 0.0 or p <= 0.0:
            return self._fallback(A, B)

        s = math.sqrt(disc)
        lam_hi = (-q + s) / (2.0 * p)
        lam_lo = (-q - s) / (2.0 * p)
        if lam_hi < lam_lo:
            lam_hi, lam_lo = lam_lo, lam_hi

        def make_vec(lam: float) -> list[float] | None:
            m00 = a - lam * d
            m01 = b - lam * e
            m11 = c - lam * f

            x0a, x1a = m01, -m00
            x0b, x1b = m11, -m01
            if abs(x0a) + abs(x1a) >= abs(x0b) + abs(x1b):
                x0, x1 = x0a, x1a
            else:
                x0, x1 = x0b, x1b

            bn2 = d * x0 * x0 + 2.0 * e * x0 * x1 + f * x1 * x1
            if not math.isfinite(bn2) or bn2 <= 0.0:
                return None
            scale = 1.0 / math.sqrt(bn2)
            return [x0 * scale, x1 * scale]

        v_hi = make_vec(lam_hi)
        v_lo = make_vec(lam_lo)
        if v_hi is None or v_lo is None:
            return self._fallback(A, B)

        return ([lam_hi, lam_lo], [v_hi, v_lo])

    def solve(self, problem, **kwargs) -> Any:
        A, B = problem
        A = self._asarray(A, dtype=float)
        B = self._asarray(B, dtype=float)
        n = A.shape[0]

        if n == 1:
            b = float(B[0, 0])
            return ([float(A[0, 0] / b)], [[1.0 / math.sqrt(b)]])

        if n == 2:
            return self._solve_2x2(A, B)

        if n <= 4:
            return self._fallback(A, B)

        if self._use_dsygvd:
            try:
                eigenvalues, eigenvectors, info = self._scipy_dsygvd(
                    np.array(A, dtype=float, order="F", copy=True),
                    np.array(B, dtype=float, order="F", copy=True),
                    itype=1,
                    jobz="V",
                    uplo="L",
                    overwrite_a=1,
                    overwrite_b=1,
                )
                if info == 0:
                    return (eigenvalues[::-1].tolist(), eigenvectors[:, ::-1].T.tolist())
            except Exception:
                self._use_dsygvd = False

        seigh = self._scipy_eigh
        if seigh is not None:
            try:
                eigenvalues, eigenvectors = seigh(
                    A,
                    B,
                    lower=True,
                    eigvals_only=False,
                    check_finite=False,
                    overwrite_a=False,
                    overwrite_b=False,
                    driver="gvd",
                )
                return (eigenvalues[::-1].tolist(), eigenvectors[:, ::-1].T.tolist())
            except Exception:
                pass

        return self._fallback(A, B)