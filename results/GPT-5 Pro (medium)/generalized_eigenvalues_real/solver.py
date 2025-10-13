from __future__ import annotations

from typing import Any
import numpy as np

# Try to import SciPy fast paths once at module import
try:
    from scipy.linalg import eigh as sp_eigh  # type: ignore
    from scipy.linalg import solve_triangular as sp_solve_triangular  # type: ignore
    from scipy.linalg import cholesky as sp_cholesky  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    sp_eigh = None
    sp_solve_triangular = None
    sp_cholesky = None
    _HAVE_SCIPY = False

class Solver:
    def solve(self, problem, **kwargs) -> list[float]:
        """
        Solve the generalized symmetric-definite eigenvalue problem:
            A x = λ B x
        where A is symmetric and B is symmetric positive definite.

        Returns eigenvalues sorted in descending order as a Python list[float].
        """
        A, B = problem  # Expect numpy arrays
        n = A.shape[0]
        if n == 0:
            return []
        if n == 1:
            # Fast path for 1x1: λ = A11 / B11
            return [float(A[0, 0] / B[0, 0])]

        # Preferred fast path: SciPy's generalized symmetric eigenvalue solver
        if _HAVE_SCIPY:
            # Use Fortran-ordered copies to avoid internal copies, allow overwrite for speed (safe on our copies).
            A_loc = np.array(A, copy=True, order="F")
            B_loc = np.array(B, copy=True, order="F")
            try:
                # Use divide-and-conquer driver for speed on full spectrum if available
                vals = sp_eigh(
                    A_loc,
                    B_loc,
                    lower=True,
                    eigvals_only=True,
                    overwrite_a=True,
                    overwrite_b=True,
                    check_finite=False,
                    driver="gvd",
                )
            except Exception:
                vals = sp_eigh(
                    A_loc,
                    B_loc,
                    lower=True,
                    eigvals_only=True,
                    overwrite_a=True,
                    overwrite_b=True,
                    check_finite=False,
                )
        else:
            # Fallback: Cholesky-based reduction to standard eigenproblem without explicit inverses
            # Prefer SciPy cholesky if available (unlikely in this branch), otherwise NumPy.
            L = sp_cholesky(B, lower=True, check_finite=False, overwrite_a=False) if sp_cholesky else np.linalg.cholesky(B)

            if sp_solve_triangular is not None:
                # Compute A_tilde = L^{-1} A L^{-T} efficiently
                Y = sp_solve_triangular(L, np.array(A, copy=True, order="F"), lower=True, check_finite=False, overwrite_b=True)
                X_T = sp_solve_triangular(L, Y.T, lower=True, check_finite=False, overwrite_b=True)
                A_tilde = X_T.T
            else:
                # Generic solve fallback
                Y = np.linalg.solve(L, A)
                X_T = np.linalg.solve(L, Y.T)
                A_tilde = X_T.T
            # Use eigvalsh since A_tilde is symmetric; returns ascending order
            vals = np.linalg.eigvalsh(A_tilde)

        # eigh/eigvalsh return ascending eigenvalues; reverse to get descending order
        vals_desc = vals[::-1].copy()
        return [float(v) for v in vals_desc]