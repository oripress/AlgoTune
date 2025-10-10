from typing import Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute optimal control sequence via backward Riccati recursion.
        Optimized with Cholesky decomposition for positive definite systems.
        """
        # Convert to numpy arrays
        A = np.asarray(problem["A"], dtype=np.float64, order='C')
        B = np.asarray(problem["B"], dtype=np.float64, order='C')
        Q = np.asarray(problem["Q"], dtype=np.float64, order='C')
        R = np.asarray(problem["R"], dtype=np.float64, order='C')
        P = np.asarray(problem["P"], dtype=np.float64, order='C')
        T = problem["T"]
        x0 = np.asarray(problem["x0"], dtype=np.float64).reshape(-1, 1)
        
        n, m = B.shape
        
        # Pre-allocate
        K = np.empty((T, m, n), dtype=np.float64)
        S = P.copy()
        
        BT = B.T
        
        # Backward Riccati recursion
        for t in range(T - 1, -1, -1):
            temp = BT @ S
            M1 = R + temp @ B
            M2 = temp @ A
            
            # Use Cholesky decomposition for faster solving
            try:
                c, low = cho_factor(M1, lower=True, check_finite=False)
                K[t] = cho_solve((c, low), M2, check_finite=False)
            except np.linalg.LinAlgError:
                K[t] = np.linalg.pinv(M1) @ M2
            
            Acl = A - B @ K[t]
            # Update S for next iteration
            S = Q + K[t].T @ R @ K[t] + Acl.T @ S @ Acl
            S = (S + S.T) * 0.5
        
        # Forward simulation
        U = np.empty((T, m), dtype=np.float64)
        x = x0
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.ravel()
            x = A @ x + B @ u
        
        return {"U": U}