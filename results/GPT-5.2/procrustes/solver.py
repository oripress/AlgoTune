from __future__ import annotations

from typing import Any

import numpy as np

_SVD = np.linalg.svd
_MATMUL = np.matmul

class Solver:
    __slots__ = ()

    def solve(self, problem: dict, **kwargs) -> Any:
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}

        # Avoid copies when possible; keep float64 for stable/fast LAPACK path.
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
            return {}
        n = A.shape[0]
        if n != A.shape[1]:
            return {}

        # M = B @ A^T, allocate M in Fortran order to reduce SVD internal copies.
        M = np.empty((n, n), dtype=np.float64, order="F")
        _MATMUL(B, A.T, out=M)

        # SVD(M) = U S V^T
        U, _, Vt = _SVD(M, full_matrices=False)

        # G = U @ V^T; reuse M buffer to avoid an extra allocation.
        _MATMUL(U, Vt, out=M)

        return {"solution": M}