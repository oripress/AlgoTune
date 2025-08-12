from typing import Any
import numpy as np
from scipy.linalg import solve, solve_discrete_are, blas

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute optimal control sequence via backward Riccati recursion.
        
        Returns dict with key "U" (shape (T, m)).
        """
        # Extract problem data and convert to numpy arrays with appropriate dtypes
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        Q = np.asarray(problem["Q"], dtype=np.float64)
        R = np.asarray(problem["R"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        T = int(problem["T"])
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        
        n, m = B.shape
        
        # Pre-allocate arrays for efficiency
        S = np.empty((T + 1, n, n), dtype=np.float64)
        K = np.empty((T, m, n), dtype=np.float64)
        S[T] = P
        
        # Backward Riccati recursion - optimized version
        BT = B.T  # Precompute B transpose
        AT = A.T  # Precompute A transpose
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            
            # Precompute common terms
            BT_St1 = BT @ St1
            BT_St1_B = BT_St1 @ B
            BT_St1_A = BT_St1 @ A
            
            # Compute feedback gain K[t] directly
            try:
                K[t] = solve(R + BT_St1_B, BT_St1_A, assume_a="pos")
            except np.linalg.LinAlgError:
                M1 = R + BT_St1_B
                K[t] = np.linalg.pinv(M1) @ BT_St1_A
            
            # Update S[t] using the algebraic Riccati equation
            Acl = A - B @ K[t]
            AclT = Acl.T
            KtT = K[t].T
            S[t] = Q + KtT @ R @ K[t] + AclT @ St1 @ Acl
            
            # Ensure symmetry
            S[t] = (S[t] + S[t].T) / 2
            
        # Forward simulation to compute control sequence
        U = np.empty((T, m), dtype=np.float64)
        x = x0.copy()
        
        for t in range(T):
            # Compute control input using precomputed feedback gains
            u = -K[t] @ x  # Apply the feedback gain with negative sign
            U[t] = u.ravel()  # This should be a 1D array
            x = A @ x + B @ u
        
        return {"U": U}