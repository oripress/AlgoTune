import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
from scipy import linalg
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = np.asarray(problem["matrix"], dtype=np.float32, order='C')
        k = problem["n_components"]
        matrix_type = problem.get("matrix_type", "")
        
        n, m = A.shape
        n_oversamples = min(2, k)
        n_random = min(k + n_oversamples, min(n, m))
        
        # Adaptive iteration count
        if matrix_type == "ill_conditioned":
            n_iter = 1
        else:
            n_iter = 0
        
        # Generate random matrix with fixed seed
        np.random.seed(42)
        Q = np.random.randn(m, n_random).astype(np.float32, order='C')
        
        # Initial projection
        Q = np.dot(A, Q)
        Q, _ = linalg.qr(Q, mode='economic', check_finite=False, overwrite_a=True)
        
        # Power iteration with QR decomposition
        for _ in range(n_iter):
            Q = np.dot(A.T, Q)
            Q, _ = linalg.qr(Q, mode='economic', check_finite=False, overwrite_a=True)
            Q = np.dot(A, Q)
            Q, _ = linalg.qr(Q, mode='economic', check_finite=False, overwrite_a=True)
        
        # Form B = Q.T @ A
        B = np.dot(Q.T, A)
        
        # SVD of B using economic mode
        U_hat, s, Vt = linalg.svd(B, full_matrices=False, check_finite=False, lapack_driver='gesdd', overwrite_a=True)
        
        # Form U
        U = np.dot(Q, U_hat)
        
        # Return first k components
        return {
            "U": U[:, :k],
            "S": s[:k],
            "V": Vt.T[:, :k]
        }