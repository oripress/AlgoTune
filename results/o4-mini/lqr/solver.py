import numpy as np
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem, **kwargs):
        # Parse and convert inputs
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        Q = np.array(problem["Q"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        P = np.array(problem["P"], dtype=np.float64)
        T = int(problem["T"])
        x0 = np.array(problem["x0"], dtype=np.float64)

        # Dimensions
        n, m = B.shape

        # Pre-allocate cost-to-go and gains
        S = np.empty((T + 1, n, n), dtype=np.float64)
        K = np.empty((T, m, n), dtype=np.float64)
        S[T] = P

        # Precompute transpose
        B_T = B.T

        # Backward Riccati recursion
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            M1 = R + B_T @ St1 @ B
            M2 = B_T @ St1 @ A
            try:
                # Solve M1 K = M2
                K[t] = linalg_solve(M1, M2, assume_a="pos")
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if ill-conditioned
                K[t] = np.linalg.pinv(M1) @ M2
            # Closed-loop system matrix
            Acl = A - B @ K[t]
            # Update cost-to-go
            S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            # Enforce symmetry
            S[t] = 0.5 * (S[t] + S[t].T)

        # Forward simulation to compute optimal controls
        U = np.empty((T, m), dtype=np.float64)
        x = x0.copy().ravel()
        for t in range(T):
            u = -K[t] @ x
            U[t] = u
            x = A @ x + B @ u

        return {"U": U}