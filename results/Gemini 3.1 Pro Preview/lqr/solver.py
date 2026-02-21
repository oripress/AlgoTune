from typing import Any
import numpy as np
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem: dict) -> dict:
        import numpy as np
        from scipy.linalg import cho_factor, cho_solve
        
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        R = problem["R"]
        P = problem["P"]
        T = problem["T"]
        x0 = problem["x0"]

        n, m = B.shape
        K = np.zeros((T, m, n))
        
        S = P
        for t in range(T - 1, -1, -1):
            B_T_S = B.T @ S
            M1 = R + B_T_S @ B
            M2 = B_T_S @ A
            try:
                c, lower = cho_factor(M1, lower=True)
                Kt = cho_solve((c, lower), M2)
            except np.linalg.LinAlgError:
                Kt = np.linalg.pinv(M1) @ M2
            K[t] = Kt
            Acl = A - B @ Kt
            S = Q + Kt.T @ R @ Kt + Acl.T @ S @ Acl
            S = (S + S.T) / 2.0

        U = np.zeros((T, m))
        x = x0
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.ravel()
            x = A @ x + B @ u

        return {"U": U.tolist()}