import numpy as np
from scipy.linalg import solve as linalg_solve

class Solver:
    def __init__(self):
        # No initialization needed
        pass

    def solve(self, problem: dict, **kwargs) -> dict:
        # Parse inputs
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        Q = np.array(problem["Q"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        P = np.array(problem["P"], dtype=np.float64)
        T = int(problem["T"])
        x0 = np.array(problem["x0"], dtype=np.float64)
        # Dimensions
        n = A.shape[0]
        m = B.shape[1]
        # Pre-allocate cost-to-go and gain matrices
        S = np.zeros((T + 1, n, n), dtype=np.float64)
        K = np.zeros((T, m, n), dtype=np.float64)
        # Terminal cost
        S[T] = P
        # Backward Riccati recursion
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            M1 = R + B.T.dot(St1).dot(B)
            M2 = B.T.dot(St1).dot(A)
            try:
                # Solve M1 K[t] = M2
                K[t] = linalg_solve(M1, M2, assume_a="pos")
            except Exception:
                K[t] = np.linalg.solve(M1, M2)
            # Closed-loop dynamics
            Acl = A - B.dot(K[t])
            # Update cost-to-go
            S[t] = Q + K[t].T.dot(R).dot(K[t]) + Acl.T.dot(St1).dot(Acl)
            # Symmetrize
            S[t] = 0.5 * (S[t] + S[t].T)
        # Forward simulation
        U = np.zeros((T, m), dtype=np.float64)
        x = x0.reshape(-1)
        for t in range(T):
            u = -K[t].dot(x)
            U[t] = u
            x = A.dot(x) + B.dot(u)
        return {"U": U.tolist()}