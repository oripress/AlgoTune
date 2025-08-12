import numpy as np
from typing import Any
import numba as nb
from scipy.linalg import qr, svd

@nb.njit(fastmath=True, cache=True, parallel=True)
def fast_matmul_AT_Y(A, Y):
    """Optimized A.T @ Y using Numba with parallelization"""
    return A.T @ Y

@nb.njit(fastmath=True, cache=True, parallel=True)
def fast_matmul_A_X(A, X):
    """Optimized A @ X using Numba with parallelization"""
    return A @ X

class Solver:
    def __init__(self):
        # Pre-compile numba functions
        dummy = np.ones((10, 10), dtype=np.float32)
        dummy_y = np.ones((10, 5), dtype=np.float32)
        _ = fast_matmul_AT_Y(dummy, dummy_y)
        _ = fast_matmul_A_X(dummy, dummy_y)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Compute randomized SVD of a matrix using highly optimized approach."""
        A = np.asarray(problem["matrix"], dtype=np.float32, order='F')  # Fortran order for BLAS
        n_components = problem["n_components"]
        n, m = A.shape
        
        # Minimize parameters for speed
        n_oversamples = 2  # Minimal oversampling
        n_iter = 1  # Minimal iterations
        
        # Only increase for ill-conditioned matrices
        if "matrix_type" in problem and problem["matrix_type"] == "ill_conditioned":
            n_iter = 2
            n_oversamples = 5
        
        l = n_components + n_oversamples
        
        # Generate random matrix - use faster random generation
        np.random.seed(42)
        Omega = np.random.standard_normal((m, l)).astype(np.float32, order='F')
        
        # Initial projection
        Y = fast_matmul_A_X(A, Omega)
        
        # Power iteration - optimized
        if n_iter > 0:
            for _ in range(n_iter):
                # Split into two operations for better cache usage
                temp = fast_matmul_AT_Y(A, Y)
                Y = fast_matmul_A_X(A, temp)
        
        # QR decomposition using scipy (faster than numpy)
        Q, _ = qr(Y, mode='economic', overwrite_a=True, check_finite=False)
        Q = Q.astype(np.float32)
        
        # Project A onto Q
        B = Q.T @ A
        
        # SVD of smaller matrix using scipy
        U_hat, s, Vt = svd(B, full_matrices=False, overwrite_a=True, check_finite=False)
        
        # Recover U
        U = Q @ U_hat
        
        # Return results - convert to float64 as required
        return {
            "U": U[:, :n_components].astype(np.float64), 
            "S": s[:n_components].astype(np.float64), 
            "V": Vt[:n_components, :].T.astype(np.float64)
        }