from typing import Any, List

import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the generalized eigenvalue problem A x = λ B x for possibly complex eigenpairs.

        Returns:
          - eigenvalues: list of complex, sorted descending by real then imaginary parts
          - eigenvectors: list of lists (length n), each a unit-norm complex eigenvector
        """
        A, B = problem

        # Prefer Fortran-contiguous float64 arrays for better LAPACK performance; avoid copy if possible
        A_f = np.asarray(A, dtype=np.float64, order="F")
        B_f = np.asarray(B, dtype=np.float64, order="F")

        # Fast path: reduce to standard eigenproblem via LU-based solving B_f X = A_f
        # Fall back to generalized eig if factorization/solve fails.
        try:
            lu, piv = la.lu_factor(B_f, overwrite_a=True, check_finite=False)
            C = la.lu_solve((lu, piv), A_f, overwrite_b=True, check_finite=False)
            eigvals, eigvecs = la.eig(
                C,
                left=False,
                right=True,
                check_finite=False,
                overwrite_a=True,
            )
        except Exception:
            # Robust fallback: generalized eigenproblem directly, using fresh Fortran-ordered copies
            A_fb = np.asarray(A, dtype=np.float64, order="F")
            B_fb = np.asarray(B, dtype=np.float64, order="F")
            eigvals, eigvecs = la.eig(
                A_fb,
                B_fb,
                left=False,
                right=True,
                check_finite=False,
                overwrite_a=True,
                overwrite_b=True,
            )

        # Normalize eigenvectors to unit norm
        norms = np.linalg.norm(eigvecs, axis=0)
        norms[norms == 0] = 1.0
        eigvecs /= norms

        # Sort by descending real part, then descending imaginary part
        order = np.lexsort((-eigvals.imag, -eigvals.real))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Convert to Python lists efficiently
        eigenvalues_list: List[complex] = eigvals.tolist()
        eigenvectors_list: List[List[complex]] = eigvecs.T.tolist()

        return (eigenvalues_list, eigenvectors_list)