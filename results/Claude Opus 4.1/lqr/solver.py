import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from typing import Any

class Solver:
    def __init__(self):
        # Pre-compile numpy operations
        pass
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute optimal control sequence via backward Riccati recursion.
        
        Returns dict with key "U" (shape (T, m)).
        """
        # Extract problem data - avoid copies where possible
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        Q = np.asarray(problem["Q"], dtype=np.float64)
        R = np.asarray(problem["R"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        T = problem["T"]
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        
        n, m = B.shape
        
        # Pre-allocate arrays
        S = np.empty((T + 1, n, n), dtype=np.float64)
        K = np.empty((T, m, n), dtype=np.float64)
        S[T] = P
        
        # Pre-compute transposes
        BT = B.T
        AT = A.T
        
        # Backward Riccati recursion with optimized operations
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            
            # Optimized matrix multiplications
            BTSt1 = BT @ St1
            M1 = R + BTSt1 @ B
            M2 = BTSt1 @ A
            
            # Use Cholesky for positive definite matrices when possible
            try:
                if m == 1:
                    # Scalar case - direct division
                    K[t] = M2 / M1[0, 0]
                else:
                    # Try Cholesky factorization for positive definite matrices
                    c, low = cho_factor(M1, lower=True, check_finite=False)
                    K[t] = cho_solve((c, low), M2, check_finite=False)
            except:
                # Fallback to standard solve
                K[t] = np.linalg.solve(M1, M2)
            
            # Compute closed-loop A matrix
            BK = B @ K[t]
            Acl = A - BK
            
            # Efficient computation of S[t]
            KT = K[t].T
            KTR = KT @ R
            AclT = Acl.T
            
            # S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            S[t] = Q + KTR @ K[t] + AclT @ St1 @ Acl
            
            # Fast symmetrization using in-place operations
            np.add(S[t], S[t].T, out=S[t])
            S[t] *= 0.5
        
        # Forward simulation with pre-allocated output
        U = np.empty((T, m), dtype=np.float64)
        
        # Use contiguous memory for x
        if x0.ndim == 1:
            x = x0.reshape(-1, 1)
        else:
            x = x0.copy()
            
        for t in range(T):
            # u = -K[t] @ x
            np.dot(K[t], x, out=U[t:t+1].T)
            U[t] *= -1
            
            # x = A @ x + B @ u
            u = U[t].reshape(-1, 1)
            x = A @ x + B @ u
        
        return {"U": U.tolist()}