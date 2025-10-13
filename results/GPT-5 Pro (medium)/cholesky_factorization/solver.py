from typing import Any

import numpy as np

# Optional SciPy import for faster LAPACK access with fewer checks
try:
    from scipy.linalg import cholesky as scipy_cholesky

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    scipy_cholesky = None
    _HAS_SCIPY = False

# Optional Numba import for a very fast small-matrix kernel
try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

def _chol_small_py(A: np.ndarray) -> np.ndarray:
    """Pure-Python (NumPy) reference Cholesky, O(n^3), lower-triangular output."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if s <= 0.0:
            # For SPD matrices, s should be > 0, but guard against tiny negatives from FP error
            s = 0.0 if s > -1e-15 else s
        Ljj = np.sqrt(s)
        L[j, j] = Ljj
        inv = 1.0 / Ljj if Ljj != 0.0 else 0.0
        for i in range(j + 1, n):
            s2 = A[i, j]
            for k in range(j):
                s2 -= L[i, k] * L[j, k]
            L[i, j] = s2 * inv
    # Zero upper triangle to be safe
    for i in range(n):
        for j2 in range(i + 1, n):
            L[i, j2] = 0.0
    return L

if _HAS_NUMBA:
    # Numba-compiled small matrix kernel
    _chol_small_jit = njit(cache=True)(_chol_small_py)
else:
    _chol_small_jit = _chol_small_py

class Solver:
    def __init__(self) -> None:
        # Pre-compile numba kernel so JIT time is not counted in solve
        if _HAS_NUMBA:
            dummy = np.zeros((1, 1), dtype=np.float64)
            _ = _chol_small_jit(dummy)

        # Threshold where our small-kernel tends to beat full LAPACK calls
        # Tune conservatively to avoid regressions on larger matrices.
        self._small_n_threshold = 32

    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the Cholesky factorization of a symmetric positive definite matrix A.

        Parameters:
            problem: dict with key "matrix" mapping to an array-like representing A.

        Returns:
            dict: {"Cholesky": {"L": L}} where L is a numpy array (lower triangular)
        """
        A = problem["matrix"]
        # Ensure numpy array of float64 without copying unless necessary
        A_np = np.asarray(A, dtype=np.float64)
        n = A_np.shape[0]

        if n <= self._small_n_threshold:
            # Fast small-size kernel (Numba if available)
            L = _chol_small_jit(A_np)
        else:
            if _HAS_SCIPY:
                # Use SciPy's LAPACK wrapper with minimal checks
                # Returns triangular matrix (lower due to lower=True)
                L = scipy_cholesky(A_np, lower=True, overwrite_a=False, check_finite=False)
            else:
                # Fallback to NumPy
                L = np.linalg.cholesky(A_np)

        return {"Cholesky": {"L": L}}