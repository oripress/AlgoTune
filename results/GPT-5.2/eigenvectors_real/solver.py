from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

class Solver:
    __slots__ = ()

    def solve(self, problem, **kwargs) -> Any:
        A = np.asarray(problem)
        n = int(A.shape[0])

        # Very small fast paths (avoid LAPACK overhead).
        if n == 1:
            return ([float(A[0, 0])], [[1.0]])

        if n == 2:
            a = float(A[0, 0])
            b = float(A[0, 1])
            d = float(A[1, 1])

            if b == 0.0:
                if a >= d:
                    return ([a, d], [[1.0, 0.0], [0.0, 1.0]])
                return ([d, a], [[0.0, 1.0], [1.0, 0.0]])

            t = 0.5 * (a + d)
            diff = 0.5 * (a - d)
            r = sqrt(diff * diff + b * b)
            lam1 = t + r
            lam2 = t - r

            x = b
            y = lam1 - a
            if abs(x) + abs(y) == 0.0:
                x = lam1 - d
                y = b
            inv_norm = 1.0 / sqrt(x * x + y * y)
            v1x = x * inv_norm
            v1y = y * inv_norm
            return ([lam1, lam2], [[v1x, v1y], [-v1y, v1x]])

        # General case: LAPACK via numpy.
        # Ensure float64; and for non-tiny matrices, prefer Fortran order to avoid internal copies.
        if A.dtype != np.float64:
            A = A.astype(np.float64, copy=False)
        if n >= 16 and A.flags["C_CONTIGUOUS"] and not A.flags["F_CONTIGUOUS"]:
            A = np.asfortranarray(A)

        w, v = np.linalg.eigh(A)  # w asc, v columns

        # Convert to Python types cheaply:
        # - build lists from contiguous arrays
        # - reverse at Python list level (avoid `tolist()` on negative-stride views)
        w_list = w.tolist()
        w_list.reverse()

        vt = np.ascontiguousarray(v.T)  # usually a no-copy view (v is typically Fortran-order)
        vecs = vt.tolist()
        vecs.reverse()

        return (w_list, vecs)