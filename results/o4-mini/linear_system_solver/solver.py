import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from typing import Any

# Pre-fetch LAPACK 'gesv' for in-place solve (minimal overhead)
try:
    from scipy.linalg.lapack import get_lapack_funcs
    _gesv, = get_lapack_funcs(('gesv',), (np.ones((1,1), dtype=np.float64),))
except Exception:
    _gesv = None

# Numba-compiled Gaussian elimination for small systems
from numba import njit

@njit(cache=True)
def _numba_solve(A, b):
    n = A.shape[0]
    # Copy for in-place
    A2 = A.copy()
    b2 = b.copy()
    # Gaussian elimination with partial pivoting
    for k in range(n):
        pivot = k
        max_val = abs(A2[k, k])
        for i in range(k+1, n):
            val = abs(A2[i, k])
            if val > max_val:
                max_val = val
                pivot = i
        if pivot != k:
            for j in range(k, n):
                tmp = A2[k, j]; A2[k, j] = A2[pivot, j]; A2[pivot, j] = tmp
            tmp = b2[k]; b2[k] = b2[pivot]; b2[pivot] = tmp
        akk = A2[k, k]
        for i in range(k+1, n):
            factor = A2[i, k] / akk
            for j in range(k+1, n):
                A2[i, j] -= factor * A2[k, j]
            b2[i] -= factor * b2[k]
    # back substitution
    x = np.empty(n, dtype=np.float64)
    for i in range(n-1, -1, -1):
        s = 0.0
        for j in range(i+1, n):
            s += A2[i, j] * x[j]
        x[i] = (b2[i] - s) / A2[i, i]
    return x

# Pre-compile numba solver on a dummy to take JIT at import time
try:
    _ = _numba_solve(np.ones((1,1), dtype=np.float64), np.ones(1, dtype=np.float64))
except Exception:
    pass

class Solver:
    def solve(self, problem: Any, **kwargs) -> list[float]:
        A = np.asarray(problem["A"], dtype=np.float64, order='F')
        b = np.asarray(problem["b"], dtype=np.float64)
        n = A.shape[0]
        if n <= 64:
            # Use Numba solver for small sizes
            return _numba_solve(A.copy(order='C'), b.copy(order='C')).tolist()
        if _gesv is not None:
            # Use LAPACK gesv: overwrite for speed
            lu, piv, x, info = _gesv(A, b, overwrite_a=True, overwrite_b=True)
            return x.tolist()
        # Fallback to NumPy
        return np.linalg.solve(A, b).tolist()