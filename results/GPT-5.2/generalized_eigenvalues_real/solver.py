from __future__ import annotations

from typing import Any

import numpy as np

try:
    # Level-3 BLAS triangular solve (fast for many RHS; avoids transpose-based solves)
    from scipy.linalg.blas import dtrsm as sp_dtrsm  # type: ignore
except Exception:  # pragma: no cover
    sp_dtrsm = None

try:
    # LAPACK-based triangular solve fallback
    from scipy.linalg import solve_triangular as sp_solve_triangular  # type: ignore
except Exception:  # pragma: no cover
    sp_solve_triangular = None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        A, B = problem

        # Ensure float64 arrays (avoid copies when possible)
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64)
        elif A.dtype != np.float64:
            A = A.astype(np.float64, copy=False)

        if not isinstance(B, np.ndarray):
            B = np.asarray(B, dtype=np.float64)
        elif B.dtype != np.float64:
            B = B.astype(np.float64, copy=False)

        n = A.shape[0]

        # Cholesky of SPD B (lower triangular)
        L = np.linalg.cholesky(B)

        # Reduce to standard symmetric eigenproblem:
        # Atilde = L^{-1} A L^{-T} computed via triangular solves (no explicit inverse)
        if sp_dtrsm is not None and n >= 16:
            # Ensure Fortran contiguity for BLAS; do not mutate A
            Lf = np.asfortranarray(L)
            X = np.array(A, dtype=np.float64, order="F", copy=True)

            # X := inv(L) * X
            X = sp_dtrsm(1.0, Lf, X, side=0, lower=1, trans_a=0, diag=0, overwrite_b=1)
            # X := X * inv(L).T = X * inv(L^T)
            X = sp_dtrsm(1.0, Lf, X, side=1, lower=1, trans_a=1, diag=0, overwrite_b=1)
            Atilde = X
        elif sp_solve_triangular is not None:
            X = sp_solve_triangular(L, A, lower=True, check_finite=False, overwrite_b=False)
            Atilde = sp_solve_triangular(
                L, X.T, lower=True, check_finite=False, overwrite_b=False
            ).T
        else:
            X = np.linalg.solve(L, A)
            Atilde = np.linalg.solve(L, X.T).T

        w = np.linalg.eigvalsh(Atilde)  # ascending
        return w[::-1].tolist()