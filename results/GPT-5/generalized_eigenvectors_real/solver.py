from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.linalg import eigh as sp_eigh
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    sp_eigh = None
    _HAVE_SCIPY = False

class Solver:
    def solve(self, problem, **kwargs) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric A and SPD B.

        Fast paths:
        - Prefer SciPy's generalized symmetric solver (scipy.linalg.eigh) with b=B, which returns
          B-orthonormal eigenvectors directly.
        - Fallback to a NumPy-only path using Cholesky and triangular solves without explicit inverses.
        """
        A_in, B_in = problem

        if _HAVE_SCIPY:
            # Prepare Fortran-ordered copies to minimize internal copies in LAPACK and avoid mutating inputs
            A = np.array(A_in, dtype=float, order="F", copy=True)
            B = np.array(B_in, dtype=float, order="F", copy=True)
            # Solve generalized symmetric eigenproblem (ascending eigenvalues)
            w, V = sp_eigh(
                A,
                B,
                lower=True,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )
            # Convert to descending order
            w = w[::-1]
            V = V[:, ::-1]
            return (w.tolist(), V.T.tolist())

        # NumPy fallback: transform to standard problem via Cholesky and triangular solves
        A = np.asarray(A_in, dtype=float)
        B = np.asarray(B_in, dtype=float)

        # Cholesky of SPD matrix B: B = L L^T
        L = np.linalg.cholesky(B)

        # Compute Atilde = L^{-1} A L^{-T} using two triangular solves
        Y = np.linalg.solve(L, A)
        Atilde = np.linalg.solve(L, Y.T).T

        # Solve standard symmetric eigenproblem (ascending eigenvalues)
        w, Z = np.linalg.eigh(np.asfortranarray(Atilde))

        # Convert to descending order
        w = w[::-1]
        Z = Z[:, ::-1]

        # Map back eigenvectors: solve L^T V = Z  => V = L^{-T} Z
        V = np.linalg.solve(L.T, Z)

        return (w.tolist(), V.T.tolist())