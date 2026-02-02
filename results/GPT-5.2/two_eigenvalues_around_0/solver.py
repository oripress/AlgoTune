from __future__ import annotations

from typing import Any

import math

import numpy as np

try:  # SciPy is available in the environment, but keep optional.
    from scipy.linalg import eigh_tridiagonal as _eigh_tridiagonal  # type: ignore
    from scipy.linalg import eigvalsh_banded as _eigvalsh_banded  # type: ignore
except Exception:  # pragma: no cover
    _eigh_tridiagonal = None
    _eigvalsh_banded = None

class Solver:
    def __init__(self) -> None:
        # Bindings for faster attribute access
        self._eigvalsh = np.linalg.eigvalsh
        self._searchsorted = np.searchsorted
        self._sqrt = math.sqrt
        self._acos = math.acos
        self._cos = math.cos
        self._pi = math.pi

        self._eigh_tridiagonal = _eigh_tridiagonal
        self._eigvalsh_banded = _eigvalsh_banded

    @staticmethod
    def _two_by_abs_from_iter(it) -> list[float]:
        best1 = 0.0
        best2 = 0.0
        a1 = float("inf")
        a2 = float("inf")
        for x in it:
            ax = -x if x < 0.0 else x
            if ax < a1:
                a2, best2 = a1, best1
                a1, best1 = ax, x
            elif ax < a2:
                a2, best2 = ax, x
        return [best1, best2]

    @staticmethod
    def _two_closest_to_zero_from_sorted(eigs: np.ndarray) -> list[float]:
        # eigs must be sorted ascending
        n = eigs.size
        i = int(np.searchsorted(eigs, 0.0))

        if i <= 0:
            x0 = float(eigs[0])
            x1 = float(eigs[1])
            if (-x0 if x0 < 0.0 else x0) <= (-x1 if x1 < 0.0 else x1):
                return [x0, x1]
            return [x1, x0]
        if i >= n:
            x0 = float(eigs[n - 1])
            x1 = float(eigs[n - 2])
            if (-x0 if x0 < 0.0 else x0) <= (-x1 if x1 < 0.0 else x1):
                return [x0, x1]
            return [x1, x0]

        best1 = 0.0
        best2 = 0.0
        a1 = float("inf")
        a2 = float("inf")

        # candidates: i-2, i-1, i, i+1 (where valid)
        if i > 1:
            x = float(eigs[i - 2])
            ax = -x if x < 0.0 else x
            if ax < a1:
                a2, best2 = a1, best1
                a1, best1 = ax, x
            elif ax < a2:
                a2, best2 = ax, x

        x = float(eigs[i - 1])
        ax = -x if x < 0.0 else x
        if ax < a1:
            a2, best2 = a1, best1
            a1, best1 = ax, x
        elif ax < a2:
            a2, best2 = ax, x

        x = float(eigs[i])
        ax = -x if x < 0.0 else x
        if ax < a1:
            a2, best2 = a1, best1
            a1, best1 = ax, x
        elif ax < a2:
            a2, best2 = ax, x

        if i + 1 < n:
            x = float(eigs[i + 1])
            ax = -x if x < 0.0 else x
            if ax < a1:
                a2, best2 = a1, best1
                a1, best1 = ax, x
            elif ax < a2:
                a2, best2 = ax, x

        return [best1, best2]

    @staticmethod
    def _maybe_diagonal_list(mat: list[list[float]]) -> bool:
        n = len(mat)
        m = 6 if n > 6 else n
        # Quick sample: if any off-diagonal in top-left block nonzero, not diagonal.
        for i in range(m):
            row = mat[i]
            for j in range(m):
                if i != j and row[j] != 0.0:
                    return False
        # Full confirmation (only reached if sample suggests diagonal).
        for i in range(n):
            row = mat[i]
            for j in range(i):
                if row[j] != 0.0:
                    return False
            for j in range(i + 1, n):
                if row[j] != 0.0:
                    return False
        return True

    def _eigvals_2x2(self, m: list[list[float]]) -> list[float]:
        a = float(m[0][0])
        b = float(m[0][1])
        c = float(m[1][1])
        tr = a + c
        det = a * c - b * b
        disc = tr * tr - 4.0 * det
        if disc < 0.0:
            disc = 0.0
        s = self._sqrt(disc)
        l1 = 0.5 * (tr - s)
        l2 = 0.5 * (tr + s)
        if (-l1 if l1 < 0.0 else l1) <= (-l2 if l2 < 0.0 else l2):
            return [l1, l2]
        return [l2, l1]

    def _eigvals_3x3_symmetric(self, m: list[list[float]]) -> list[float]:
        a00 = float(m[0][0])
        a01 = float(m[0][1])
        a02 = float(m[0][2])
        a11 = float(m[1][1])
        a12 = float(m[1][2])
        a22 = float(m[2][2])

        p1 = a01 * a01 + a02 * a02 + a12 * a12
        if p1 == 0.0:
            return self._two_by_abs_from_iter((a00, a11, a22))

        q = (a00 + a11 + a22) / 3.0
        b00 = a00 - q
        b11 = a11 - q
        b22 = a22 - q
        p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1
        p = self._sqrt(p2 / 6.0)

        invp = 1.0 / p
        b00 *= invp
        b11 *= invp
        b22 *= invp
        b01 = a01 * invp
        b02 = a02 * invp
        b12 = a12 * invp

        detb = (
            b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02)
        )
        r = 0.5 * detb

        if r <= -1.0:
            phi = self._pi / 3.0
        elif r >= 1.0:
            phi = 0.0
        else:
            phi = self._acos(r) / 3.0

        two_p = 2.0 * p
        eig1 = q + two_p * self._cos(phi)
        eig3 = q + two_p * self._cos(phi + (2.0 * self._pi / 3.0))
        eig2 = 3.0 * q - eig1 - eig3

        return self._two_by_abs_from_iter((eig1, eig2, eig3))

    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat = problem["matrix"]
        n = len(mat)

        # Tiny sizes: avoid LAPACK overhead
        if n == 2:
            return self._eigvals_2x2(mat)
        if n == 3:
            return self._eigvals_3x3_symmetric(mat)

        # Exact diagonal fast path (avoids NumPy conversion + LAPACK)
        if self._maybe_diagonal_list(mat):
            return self._two_by_abs_from_iter((float(mat[i][i]) for i in range(n)))

        a = np.asarray(mat, dtype=np.float64)

        # Try to exploit exact small bandwidth (many generators use banded operators).
        # We only attempt to recognize small bandwidths; otherwise fall back to dense.
        max_bw = 4
        bw = 0

        # Determine if there are any nonzeros on diagonals within max_bw.
        # For symmetric matrices, checking upper diagonals is sufficient.
        for k in range(1, max_bw + 1):
            if np.any(a.diagonal(k)):
                bw = k

        # Confirm that all diagonals beyond max_bw are zero; if not, treat as dense.
        is_small_banded = True
        for k in range(max_bw + 1, n):
            if np.any(a.diagonal(k)):
                is_small_banded = False
                break

        if is_small_banded and bw == 1 and self._eigh_tridiagonal is not None:
            # Tridiagonal case: even faster specialized routine.
            d = np.ascontiguousarray(a.diagonal(0))
            e = np.ascontiguousarray(a.diagonal(1))
            eigs = self._eigh_tridiagonal(d, e, eigvals_only=True, check_finite=False)
            return self._two_closest_to_zero_from_sorted(eigs)

        if is_small_banded and bw > 0 and self._eigvalsh_banded is not None:
            # Banded symmetric eigenvalues (much faster than dense when bw << n).
            # Build upper-banded storage for SciPy: shape (bw+1, n),
            # row bw is main diagonal, row bw-k is k-th superdiagonal.
            ab = np.zeros((bw + 1, n), dtype=np.float64)
            for k in range(bw + 1):
                ab[bw - k, k:] = a.diagonal(k)
            eigs = self._eigvalsh_banded(ab, lower=False, check_finite=False)
            return self._two_closest_to_zero_from_sorted(eigs)

        # Dense fallback
        eigs = self._eigvalsh(a)
        return self._two_closest_to_zero_from_sorted(eigs)