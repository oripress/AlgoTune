import numpy as np
from typing import Any, List

# Prefer SciPy's high-level routines when available.
try:
    import scipy.linalg as sla  # type: ignore
    from scipy.linalg import solve_triangular  # type: ignore
    _HAS_SCIPY = True
except Exception:
    sla = None  # type: ignore
    solve_triangular = None  # type: ignore
    _HAS_SCIPY = False

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[float]:
        """
        Solve the generalized eigenvalue problem A x = lambda B x for symmetric A and SPD B.
        Returns eigenvalues sorted in descending order.

        Fast path:
          - If SciPy is available, call scipy.linalg.eigh with Fortran-ordered input,
            eigvals_only=True, overwrite flags and check_finite=False to minimize copies
            and maximize LAPACK performance.

        Fallbacks:
          - Robust Cholesky-based reduction (L^{-1} A L^{-T}) using triangular solves.
          - As a last resort, compute B^{-1} A (or pseudo-inverse) and eigvalsh of the symmetric part.
        """
        if not (isinstance(problem, (tuple, list)) and len(problem) == 2):
            raise ValueError("Expected problem to be a tuple/list (A, B).")
        A_in, B_in = problem

        # Ensure numpy arrays with float64
        A = np.asarray(A_in, dtype=np.float64)
        B = np.asarray(B_in, dtype=np.float64)

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2-dimensional arrays.")
        if A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError("A and B must be square matrices of the same shape.")

        n = A.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [float(A[0, 0] / B[0, 0])]

        # Symmetrize (small cost relative to eigensolver)
        A = 0.5 * (A + A.T)
        B = 0.5 * (B + B.T)

        # Fast path: use SciPy's optimized solver with Fortran-ordered arrays to avoid copies.
        if _HAS_SCIPY and sla is not None:
            try:
                # Create Fortran-ordered views/copies if necessary (avoids internal copies in LAPACK wrapper).
                Af = np.array(A, dtype=np.float64, order="F", copy=False)
                Bf = np.array(B, dtype=np.float64, order="F", copy=False)
                vals = sla.eigh(Af, Bf, eigvals_only=True, overwrite_a=True, overwrite_b=True, check_finite=False)
                # eigh returns ascending eigenvalues; reverse for descending
                return [float(v) for v in vals[::-1]]
            except Exception:
                # If SciPy fails for any reason, fall through to robust numpy-based approach.
                pass

        # Robust Cholesky-based reduction using triangular solves (avoids explicit inverses).
        L = None
        try:
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            # Add progressive jitter scaled to matrix norm
            normB = np.linalg.norm(B, ord=np.inf)
            jitter = max(1e-16, 1e-12 * max(1.0, normB))
            for _ in range(16):
                try:
                    L = np.linalg.cholesky(B + np.eye(n) * jitter)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0

        if L is None:
            # Last resort: compute B^{-1} A (or use pseudo-inverse) and symmetrize
            try:
                M = np.linalg.solve(B, A)
            except np.linalg.LinAlgError:
                M = np.linalg.pinv(B) @ A
            M = 0.5 * (M + M.T)
            vals = np.linalg.eigvalsh(M)
            return [float(v) for v in vals[::-1]]

        # Use scipy.linalg.solve_triangular if available (calls BLAS/LAPACK efficiently)
        if _HAS_SCIPY and solve_triangular is not None:
            # Solve L * Y = A  => Y = L^{-1} A
            Y = solve_triangular(L, A, lower=True, trans="N", overwrite_b=False, check_finite=False)
            # Atilde = L^{-1} A L^{-T} = solve(L, Y.T).T
            Atilde = solve_triangular(L, Y.T, lower=True, trans="T", overwrite_b=False, check_finite=False).T
        else:
            Y = np.linalg.solve(L, A)
            Atilde = np.linalg.solve(L, Y.T).T

        # Ensure symmetry and compute eigenvalues
        Atilde = 0.5 * (Atilde + Atilde.T)
        vals = np.linalg.eigvalsh(Atilde)
        return [float(v) for v in vals[::-1]]