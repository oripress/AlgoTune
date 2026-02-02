from __future__ import annotations

import cmath
from functools import lru_cache
from typing import Any

import numpy as np

_EIGVALS = np.linalg.eigvals
_EIGVALSH = np.linalg.eigvalsh
_F64 = np.float64
_F32 = np.float32

@lru_cache(maxsize=128)
def _tril_idx(n: int):
    return np.tril_indices(n, -1)

@lru_cache(maxsize=128)
def _triu_idx(n: int):
    return np.triu_indices(n, 1)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute eigenvalues of a real square matrix and return them sorted
        descending by (real part, imaginary part).
        """
        a = np.asarray(problem)
        n = int(a.shape[0])

        # Fast path: 1x1
        if n == 1:
            return [complex(a[0, 0])]

        # Fast path: 2x2 closed-form (scaled + cancellation-avoiding)
        if n == 2:
            a00 = float(a[0, 0])
            a01 = float(a[0, 1])
            a10 = float(a[1, 0])
            a11 = float(a[1, 1])

            scale = max(abs(a00), abs(a01), abs(a10), abs(a11))
            if scale == 0.0:
                return [0j, 0j]

            b00 = a00 / scale
            b01 = a01 / scale
            b10 = a10 / scale
            b11 = a11 / scale

            tr = b00 + b11
            det = b00 * b11 - b01 * b10
            s = cmath.sqrt(tr * tr - 4.0 * det)

            if tr >= 0.0:
                w0n = 0.5 * (tr + s)
            else:
                w0n = 0.5 * (tr - s)

            if w0n != 0:
                w1n = det / w0n
            else:
                w1n = 0.5 * (tr - s) if tr >= 0.0 else 0.5 * (tr + s)

            w0 = w0n * scale
            w1 = w1n * scale

            if (w0.real < w1.real) or (w0.real == w1.real and w0.imag < w1.imag):
                w0, w1 = w1, w0
            return [w0, w1]

        # Normalize dtype for faster scalar ops + LAPACK.
        dt = a.dtype
        if dt != _F64 and dt != _F32:
            a = a.astype(_F64, copy=False)

        # Cheap structure heuristics (avoid O(n^2) checks unless likely).
        # 1) Upper triangular => eigenvalues are diag(a)
        # sample a few below-diagonal entries
        if (
            a[1, 0] == 0
            and a[n - 1, 0] == 0
            and a[n - 1, n - 2] == 0
            and (n < 3 or (a[2, 0] == 0 and a[2, 1] == 0))
        ):
            li = _tril_idx(n)
            if not np.any(a[li]):
                d = np.diag(a).copy()
                d.sort()
                return d[::-1].tolist()

        # 2) Lower triangular => eigenvalues are diag(a)
        if (
            a[0, 1] == 0
            and a[0, n - 1] == 0
            and a[n - 2, n - 1] == 0
            and (n < 3 or (a[0, 2] == 0 and a[1, 2] == 0))
        ):
            ui = _triu_idx(n)
            if not np.any(a[ui]):
                d = np.diag(a).copy()
                d.sort()
                return d[::-1].tolist()

        # 3) Symmetric => use eigvalsh (real, faster); only if a few entries match
        if (
            a[0, 1] == a[1, 0]
            and a[0, n - 1] == a[n - 1, 0]
            and a[1, n - 1] == a[n - 1, 1]
            and (n < 4 or a[1, 2] == a[2, 1])
        ):
            # Confirm exactly (rarely triggered due to heuristic)
            if np.array_equal(a, a.T):
                w = _EIGVALSH(a)  # ascending real
                return w[::-1].tolist()

        # General path
        w = _EIGVALS(a)
        w.sort()  # ascending by (real, imag) for complex dtypes
        return w[::-1].tolist()