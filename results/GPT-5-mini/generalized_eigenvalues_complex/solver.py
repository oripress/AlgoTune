from typing import Any, Tuple, List
import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem: Tuple[Any, Any], **kwargs) -> List[complex]:
        """
        Solve the generalized eigenvalue problem A x = lambda B x.

        Performance-minded strategy:
        - Use np.asarray to avoid unnecessary copies for the common fast path.
        - First attempt the fast numpy path: X = solve(B, A) and eigenvalues of X.
        - Only if that fails (singular B or numerical issues), create Fortran-ordered copies
          and use SciPy's Cholesky reduction or generalized solver with overwrite flags.
        - Final fallback uses pseudoinverse.
        - Sort eigenvalues by real part (desc), then imaginary part (desc).
        """
        try:
            A_in, B_in = problem
        except Exception as e:
            raise ValueError("Problem must be a pair (A, B)") from e

        # Light-weight view copies; dtype coerced to float64 if needed.
        A = np.asarray(A_in, dtype=np.float64)
        B = np.asarray(B_in, dtype=np.float64)

        # Validate shapes
        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError("A and B must be square matrices of the same shape")

        n = A.shape[0]
        if n == 0:
            return []

        eigenvalues = None

        # Attempt the fast path without forcing Fortran order or extra copies.
        try:
            # Solve B X = A for X; this will raise LinAlgError if B is singular.
            X = np.linalg.solve(B, A)
            ev = np.linalg.eigvals(X)
            if np.isfinite(ev).all():
                eigenvalues = ev
        except np.linalg.LinAlgError:
            eigenvalues = None
        except Exception:
            # Any unexpected numerical issue -> fallback to robust routines.
            eigenvalues = None

        # If the fast path didn't work, prepare Fortran-contiguous float64 copies
        # for LAPACK-backed SciPy routines (they can work in-place and be faster then).
        if eigenvalues is None:
            # Cheap symmetry check for B to decide whether Cholesky reduction is worth trying.
            symmetric_B = False
            try:
                symmetric_B = np.allclose(B, B.T, atol=1e-12, rtol=1e-8)
            except Exception:
                symmetric_B = False

            # Create Fortran-ordered copies once for SciPy routines.
            Af = np.array(A, dtype=np.float64, order="F", copy=True)
            Bf = np.array(B, dtype=np.float64, order="F", copy=True)

            if symmetric_B:
                try:
                    L = la.cholesky(Bf, lower=True, check_finite=False)
                    # C = inv(L) @ Af @ inv(L.T) computed via triangular solves
                    Y = la.solve_triangular(L, Af, trans=0, lower=True, check_finite=False)
                    Z = la.solve_triangular(L, Y.T, trans=0, lower=True, check_finite=False)
                    C = Z.T
                    ev = np.linalg.eigvals(C)
                    if np.isfinite(ev).all():
                        eigenvalues = ev
                except Exception:
                    eigenvalues = None

            # If still None, use SciPy generalized solver (QZ) allowing overwrites.
            if eigenvalues is None:
                try:
                    ev = la.eigvals(Af, Bf, overwrite_a=True, overwrite_b=True, check_finite=False)
                    if np.isfinite(ev).all():
                        eigenvalues = ev
                except Exception:
                    eigenvalues = None

        # Last-resort fallback: pseudoinverse-based standard eigenproblem.
        if eigenvalues is None:
            X = np.linalg.pinv(B).dot(A)
            ev = np.linalg.eigvals(X)
            eigenvalues = ev

        ev = np.asarray(eigenvalues, dtype=np.complex128)
 
        # Sort: primary by real part descending, secondary by imag part descending.
        if ev.size > 1:
            # Single lexsort: primary -real (descending), secondary -imag (descending).
            # np.lexsort uses the last key as the primary key and sorts ascending,
            # hence we pass (-ev.imag, -ev.real).
            idx = np.lexsort((-ev.imag, -ev.real))
        else:
            idx = np.arange(ev.size)

        ev_sorted = ev[idx]

        # Defensive: ensure length is exactly n
        if ev_sorted.size != n:
            if ev_sorted.size > n:
                ev_sorted = ev_sorted[:n]
            else:
                pad = np.zeros(n - ev_sorted.size, dtype=np.complex128)
                ev_sorted = np.concatenate((ev_sorted, pad))

        return ev_sorted.tolist()