import numpy as np
import scipy.linalg
from typing import Any
from numba import njit

@njit(cache=True)
def solve_riccati(A, B, max_iter=1000, tol=1e-5):
    n = A.shape[0]
    m = B.shape[1]
    
    P = np.eye(n)
    I_m = np.eye(m)
    AT = A.T
    BT = B.T
    
    for _ in range(max_iter):
        BTP = BT @ P
        M1 = I_m + BTP @ B
        M2 = BTP @ A
        
        temp = np.linalg.inv(M1) @ M2
        
        ATP = AT @ P
        P_new = ATP @ (A - B @ temp)
        for i in range(n):
            P_new[i, i] += 1.0
        
        # Check convergence
        max_diff = 0.0
        for i in range(n):
            for j in range(n):
                diff = abs(P[i, j] - P_new[i, j])
                if diff > max_diff:
                    max_diff = diff
                    
        if max_diff < tol:
            K = -temp
            return K, P_new, True
        P = P_new
        
    return np.zeros((m, n)), P, False

class Solver:
    def __init__(self):
        # Trigger Numba compilation
        A = np.eye(2)
        B = np.eye(2)
        solve_riccati(A, B)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        
        try:
            K, P, converged = solve_riccati(A, B)
            if converged:
                return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
            
            # Fallback to DARE
            n, m = A.shape[0], B.shape[1]
            Q = np.eye(n)
            R = np.eye(m)
            P = scipy.linalg.solve_discrete_are(A, B, Q, R)
            K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}