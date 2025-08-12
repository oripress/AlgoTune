# solver.py optimized with Numba-JIT Jacobi for small matrices and LAPACK for large ones
import numpy as np
from typing import Any, List, Tuple

# Optional SciPy for faster LAPACK drivers on large problems.
try:
    import scipy.linalg as sla  # type: ignore
except Exception:
    sla = None

# Optional Numba for fast small-matrix eigensolver (Jacobi).
NUMBA_AVAILABLE = False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _jacobi_eig_inplace(A):
        """
        In-place Jacobi eigenvalue algorithm for symmetric matrix A.
        Returns (w, V) where columns of V are eigenvectors and w are eigenvalues.
        """
        n = A.shape[0]
        V = np.eye(n, dtype=np.float64)
        tol = 1e-12
        max_sweeps = 5 * n * n
        for sweep in range(max_sweeps):
            # Find largest off-diagonal element
            max_val = 0.0
            p = 0
            q = 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    aij = abs(A[i, j])
                    if aij > max_val:
                        max_val = aij
                        p = i
                        q = j
            if max_val <= tol:
                break
            apq = A[p, q]
            app = A[p, p]
            aqq = A[q, q]
            # Compute rotation
            tau = (aqq - app) / (2.0 * apq)
            if tau >= 0.0:
                t = 1.0 / (tau + (1.0 + tau * tau) ** 0.5)
            else:
                t = -1.0 / (-tau + (1.0 + tau * tau) ** 0.5)
            c = 1.0 / (1.0 + t * t) ** 0.5
            s = t * c
            # Rotate A
            for i in range(n):
                if i != p and i != q:
                    aip = A[i, p]
                    aiq = A[i, q]
                    A[i, p] = aip * c - aiq * s
                    A[p, i] = A[i, p]
                    A[i, q] = aiq * c + aip * s
                    A[q, i] = A[i, q]
            A[p, p] = app - t * apq
            A[q, q] = aqq + t * apq
            A[p, q] = 0.0
            A[q, p] = 0.0
            # Update eigenvector matrix V
            for i in range(n):
                vip = V[i, p]
                viq = V[i, q]
                V[i, p] = vip * c - viq * s
                V[i, q] = viq * c + vip * s
        # Extract eigenvalues
        w = np.empty(n, dtype=np.float64)
        for i in range(n):
            w[i] = A[i, i]
        return w, V

    # Force compilation at import time (small dummy array). Compilation time is not counted in solve runtime.
    try:
        _ = _jacobi_eig_inplace(np.zeros((3, 3), dtype=np.float64))
    except Exception:
        # If compilation or initial run fails, mark Numba as unavailable at runtime.
        NUMBA_AVAILABLE = False

class Solver:
    def solve(self, problem: Any, **kwargs) -> Tuple[List[float], List[List[float]]]:
        """
        Compute eigenvalues and orthonormal eigenvectors for a real symmetric matrix.

        Uses:
          - Numba-JIT Jacobi solver for small matrices (cheap per-call overhead).
          - SciPy LAPACK (driver='evd') if available for large matrices.
          - numpy.linalg.eigh as fallback.
        """
        A = np.array(problem, dtype=np.float64, copy=False, order='C')

        # Basic validations and trivial cases
        if A.ndim == 0:
            return [float(A)], [[1.0]]
        if A.ndim == 1:
            if A.size == 1:
                return [float(A[0])], [[1.0]]
            raise ValueError("Input must be a 2D square matrix.")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square 2D array-like matrix.")

        n = A.shape[0]
        if n == 0:
            return [], []
        if n == 1:
            return [float(A[0, 0])], [[1.0]]

        # Hard-coded 2x2 analytic solver to avoid overhead for tiny matrices.
        if n == 2:
            a = float(A[0, 0])
            b = 0.5 * (float(A[0, 1]) + float(A[1, 0]))
            d = float(A[1, 1])
            eps = np.finfo(float).eps
            if abs(b) <= 16 * eps * (1.0 + abs(a) + abs(d)):
                if a >= d:
                    return [a, d], [[1.0, 0.0], [0.0, 1.0]]
                else:
                    return [d, a], [[0.0, 1.0], [1.0, 0.0]]
            t = 0.5 * (a + d)
            delta = 0.5 * (a - d)
            s = np.hypot(delta, b)
            lam1 = t + s
            lam2 = t - s
            v1 = np.array([b, lam1 - a], dtype=np.float64)
            normv1 = np.linalg.norm(v1)
            if normv1 < 1e-20:
                v1 = np.array([lam1 - d, b], dtype=np.float64)
                normv1 = np.linalg.norm(v1)
            if normv1 < 1e-20:
                v1 = np.array([1.0, 0.0], dtype=np.float64)
                normv1 = 1.0
            v1 /= normv1
            v2 = np.array([-v1[1], v1[0]], dtype=np.float64)
            return [float(lam1), float(lam2)], [v1.tolist(), v2.tolist()]

        # Thresholds to select method
        JACOBI_MAX_N = 64  # Use Numba Jacobi for n <= this when available
        SCIPY_THRESHOLD = 128  # Prefer SciPy driver for n >= this when available

        # Use Numba Jacobi for small matrices to reduce per-call overhead.
        if NUMBA_AVAILABLE and n <= JACOBI_MAX_N:
            Ac = np.array(A, dtype=np.float64, copy=True)  # contiguous copy for numba
            try:
                w, V = _jacobi_eig_inplace(Ac)
                # Sort descending
                idx = np.argsort(w)[::-1]
                w = w[idx]
                V = V[:, idx]
                return [float(x) for x in w.tolist()], V.T.tolist()
            except Exception:
                # Fall through to LAPACK-based solvers if numba solver fails.
                pass

        # For larger matrices, prefer SciPy's driver if available.
        try:
            if sla is not None and n >= SCIPY_THRESHOLD:
                try:
                    w, v = sla.eigh(A, overwrite_a=False, check_finite=False, driver="evd")
                except TypeError:
                    w, v = sla.eigh(A, overwrite_a=False, check_finite=False)
            else:
                w, v = np.linalg.eigh(A)
        except Exception:
            # Robust fallback: make Fortran-ordered copy and use NumPy's eigh.
            Af = np.asfortranarray(A, dtype=np.float64)
            w, v = np.linalg.eigh(Af)

        # Convert to descending order and return eigenvectors as rows.
        w = w[::-1]
        v = v[:, ::-1]
        return [float(x) for x in w.tolist()], v.T.tolist()