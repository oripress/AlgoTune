from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> list[complex]:
        """
        Solve the generalized eigenvalue problem A x = λ B x and return eigenvalues
        sorted in descending order by real part, then by imaginary part.
        """
        A_in, B_in = problem

        # Use float64 for stability and Fortran order to minimize internal copies in LAPACK.
        A = np.array(A_in, dtype=np.float64, order="F", copy=False)
        B = np.array(B_in, dtype=np.float64, order="F", copy=False)

        n = A.shape[0] if A.ndim == 2 else 0
        if n == 0:
            return []

        if n == 1:
            b = B[0, 0]
            a = A[0, 0]
            if b != 0.0:
                w = np.array([a / b], dtype=np.complex128)
            else:
                # Fall back to robust solver for degenerate cases.
                w = la.eigvals(A, B, overwrite_a=True, check_finite=False)
        elif n == 2:
            # Closed-form solution via quadratic for det(A - λ B) = 0
            a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
            e, f, g, h = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
            a0 = a * d - b * c  # det(A)
            a2 = e * h - f * g  # det(B)
            a1 = -(a * h + d * e - b * g - c * f)
            # Handle degenerate cases robustly
            if a2 == 0.0:
                if a1 == 0.0:
                    # Fall back to robust solver
                    w = la.eigvals(A, B, overwrite_a=True, check_finite=False)
                else:
                    w = np.array([-a0 / a1, -a0 / a1], dtype=np.complex128)
            else:
                disc = a1 * a1 - 4.0 * a2 * a0
                sdisc = np.lib.scimath.sqrt(disc)
                w1 = (-a1 + sdisc) / (2.0 * a2)
                w2 = (-a1 - sdisc) / (2.0 * a2)
                w = np.array([w1, w2], dtype=np.complex128)
        elif n <= 64:
            # Fast path: reduce to standard eigenproblem using LU (if well-conditioned).
            try:
                lu, piv = la.lu_factor(B, check_finite=False)
                diagU = np.abs(np.diag(lu))
                normB = np.linalg.norm(B)
                if normB == 0.0 or diagU.min(initial=np.inf) <= 1e-15 * max(1.0, normB):
                    raise la.LinAlgError("Singular or ill-conditioned B")
                M = la.lu_solve((lu, piv), A, check_finite=False)
                w = np.linalg.eigvals(M)
            except Exception:
                w = la.eigvals(A, B, overwrite_a=True, check_finite=False)
        else:
            # General path for larger matrices.
            w = la.eigvals(A, B, overwrite_a=True, check_finite=False)

        # Sort by descending real part, then descending imaginary part.
        order = np.lexsort((-w.imag, -w.real))
        w_sorted = w[order]

        return w_sorted.tolist()