import numpy as np
from numpy.random import RandomState
from typing import Any
from scipy.linalg import svd as scipy_svd, qr as scipy_qr, lu as scipy_lu, blas

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        A = np.asarray(problem["matrix"], dtype=np.float64)
        n_components = problem["n_components"]
        matrix_type = problem.get("matrix_type", "default")
        n, m = A.shape
        
        k = n_components
        
        # Tune iterations per matrix type - minimize while staying valid
        if matrix_type == "ill_conditioned":
            n_iter = 5
        elif matrix_type == "low_rank":
            n_iter = 1
        elif matrix_type == "sparse":
            n_iter = 3
        else:
            n_iter = 3
        
        # oversampling parameter
        p = min(10, min(n, m) - k)
        if p < 2:
            p = max(0, min(n, m) - k)
        
        total = k + p
        
        # If matrix is small enough, just do truncated via full SVD
        if min(n, m) <= max(2 * total + 10, 50):
            U_full, s_full, Vt_full = scipy_svd(A, full_matrices=False, 
                                                  check_finite=False, lapack_driver='gesdd')
            return {"U": U_full[:, :k], "S": s_full[:k], "V": Vt_full[:k, :].T}
        
        rng = RandomState(42)
        
        # Transpose approach: work on the smaller dimension
        transpose = n < m
        if transpose:
            A = A.T
            n, m = m, n
        
        # Use float32 for power iteration phase (2x faster BLAS)
        A32 = A.astype(np.float32)
        Omega = rng.standard_normal((m, total)).astype(np.float32)
        Y = A32 @ Omega  # (n, total) in float32
        
        # Power iteration with LU normalization in float32
        for i in range(n_iter):
            Z = A32.T @ Y  # (m, total)
            Z, _ = scipy_lu(Z, permute_l=True, check_finite=False)
            Y = A32 @ Z    # (n, total)
            if i < n_iter - 1:
                Y, _ = scipy_lu(Y, permute_l=True, check_finite=False)
        
        # Switch back to float64 for final computation
        Y = Y.astype(np.float64)
        
        # Final QR for orthogonality
        Q, _ = scipy_qr(Y, mode='economic', check_finite=False)
        
        # Form small matrix and SVD in float64
        B = Q.T @ A  # (total, m)
        U_B, s_B, Vt_B = scipy_svd(B, full_matrices=False, check_finite=False, 
                                     lapack_driver='gesdd')
        
        U = Q @ U_B[:, :k]
        s = s_B[:k]
        V = Vt_B[:k, :].T
        
        if transpose:
            return {"U": V, "S": s, "V": U}
        return {"U": U, "S": s, "V": V}