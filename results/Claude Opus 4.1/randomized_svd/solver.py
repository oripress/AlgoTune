import numpy as np
from typing import Any
from scipy.linalg import svd, qr
import numba as nb

@nb.jit(nopython=True, cache=True, fastmath=True, parallel=True)
def fast_matmul(A, B):
    """Ultra-fast matrix multiplication with numba."""
    return np.dot(A, B)

@nb.jit(nopython=True, cache=True, fastmath=True)
def power_iter_numba(A, Y, n_iter):
    """Optimized power iterations."""
    for _ in range(n_iter):
        Y = np.dot(A, np.dot(A.T, Y))
    return Y

class Solver:
    def __init__(self):
        # Precompile numba functions
        dummy = np.ones((2, 2), dtype=np.float64)
        fast_matmul(dummy, dummy)
        power_iter_numba(dummy, dummy, 1)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Hybrid optimized randomized SVD."""
        A = np.asarray(problem["matrix"], dtype=np.float64, order='C')
        n_components = problem["n_components"]
        matrix_type = problem.get("matrix_type", "standard")
        
        n, m = A.shape
        k = n_components
        
        # Minimal oversampling - just k+1 for speed
        p = min(k + 1, min(n, m))
        
        # Generate random matrix
        np.random.seed(42)
        Omega = np.random.randn(m, p).astype(np.float64, order='C')
        
        # Fast matrix multiplication with numba
        Y = fast_matmul(A, Omega)
        
        # Only do power iterations for ill-conditioned matrices
        if matrix_type == "ill_conditioned":
            Y = power_iter_numba(A, Y, 10)
        elif n_iter := (1 if n*m > 10000 else 0):
            # Single iteration for large matrices
            Y = power_iter_numba(A, Y, n_iter)
        
        # QR decomposition - fastest mode
        Q, _ = qr(Y, mode='economic', overwrite_a=True, check_finite=False)
        
        # Project A onto Q
        B = fast_matmul(Q.T, A)
        
        # SVD of smaller matrix
        U_tilde, s, Vt = svd(B, full_matrices=False, overwrite_a=True, check_finite=False)
        
        # Transform back
        U = fast_matmul(Q, U_tilde)
        
        # Keep only k components - use contiguous slices
        return {
            "U": np.ascontiguousarray(U[:, :k]),
            "S": np.ascontiguousarray(s[:k]),
            "V": np.ascontiguousarray(Vt[:k, :].T)
        }