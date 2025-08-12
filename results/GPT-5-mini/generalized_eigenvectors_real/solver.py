import numpy as np
from typing import Any, List, Optional

# Prefer SciPy's optimized routines when available
try:
    import scipy.linalg as _sla
except Exception:
    _sla = None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the generalized eigenproblem A x = lambda B x for symmetric A and SPD B.

        Returns:
            (eigenvalues, eigenvectors)
        where eigenvalues is a list of floats sorted in descending order and
        eigenvectors is a list of lists (each vector normalized w.r.t. the B-inner product).
        """
        A, B = problem

        # Ensure numpy arrays (avoiding unnecessary copies when possible)
        A = np.array(A, dtype=np.float64, copy=False)
        B = np.array(B, dtype=np.float64, copy=False)

        # Basic shape checks
        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError("A and B must be square matrices of the same shape.")
        n = A.shape[0]
        if n == 0:
            return ([], [])

        # Fast path for 1x1 problems
        if n == 1:
            a = float(A[0, 0])
            b = float(B[0, 0])
            if b > 0 and np.isfinite(b):
                v0 = 1.0 / np.sqrt(b)
            else:
                v0 = 1.0
            return ([a / b], [[float(v0)]])

        # Preferred path: use SciPy's generalized symmetric eigensolver (if available)
        if _sla is not None:
            try:
                # check_finite=False avoids an extra copy; overwrite_a/b=True allows in-place work
                evals, evecs = _sla.eigh(
                    A,
                    B,
                    eigvals_only=False,
                    check_finite=False,
                    overwrite_a=True,
                    overwrite_b=True,
                    lower=True,
                )
                # Convert to descending order
                evals = evals[::-1]
                evecs = evecs[:, ::-1]

                eigenvalues_list: List[float] = [float(v) for v in evals]
                # eigenvectors are columns; return each as a list (B-normalized by construction)
                eigenvectors_list: List[List[float]] = [list(map(float, vec)) for vec in evecs.T]
                return (eigenvalues_list, eigenvectors_list)
            except Exception:
                # If SciPy fails for some reason, fall back to the Cholesky-based approach below
                pass

        # Fallback: transform to standard eigenproblem via Cholesky of B.
        try:
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            # Regularize slightly if B is numerically singular (rare given problem statement)
            jitter = 1e-12
            L = np.linalg.cholesky(B + jitter * np.eye(n, dtype=np.float64))

        # Solve triangular systems efficiently if possible
        if _sla is not None:
            try:
                from scipy.linalg import solve_triangular as _solve_triangular

                # Y = inv(L) @ A  -> solve L Y = A
                Y = _solve_triangular(L, A, lower=True, trans='N', overwrite_b=False, check_finite=False)
                # Atilde = inv(L) @ A @ inv(L).T  -> solve L Z = Y.T, then transpose
                Atilde = _solve_triangular(L, Y.T, lower=True, trans='N', overwrite_b=False, check_finite=False).T
            except Exception:
                # If triangular solver isn't available for some reason, use general solves
                Y = np.linalg.solve(L, A)
                Atilde = np.linalg.solve(L, Y.T).T
        else:
            Y = np.linalg.solve(L, A)
            Atilde = np.linalg.solve(L, Y.T).T

        # Enforce symmetry to reduce numerical asymmetry
        Atilde = (Atilde + Atilde.T) * 0.5

        # Solve standard symmetric eigenproblem
        evals, yvecs = np.linalg.eigh(Atilde)

        # Reverse to descending order and map back eigenvectors: x = inv(L).T @ y
        evals = evals[::-1]
        yvecs = yvecs[:, ::-1]
        xvecs = np.linalg.solve(L.T, yvecs)

        eigenvalues_list: List[float] = [float(v) for v in evals]
        eigenvectors_list: List[List[float]] = [list(map(float, vec)) for vec in xvecs.T]

        return (eigenvalues_list, eigenvectors_list)