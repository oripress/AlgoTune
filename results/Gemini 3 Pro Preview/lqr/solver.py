from typing import Any
import numpy as np
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A, B = problem["A"], problem["B"]
        Q, R, P = problem["Q"], problem["R"], problem["P"]
        T, x0 = problem["T"], problem["x0"]

        # Ensure inputs are numpy arrays
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        P = np.asarray(P, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1, 1)

        n, m = B.shape
        S = np.zeros((T + 1, n, n))
        K = np.zeros((T, m, n))
        S[T] = P

        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            M1 = R + B.T @ St1 @ B
            M2 = B.T @ St1 @ A
            try:
                K[t] = linalg_solve(M1, M2, assume_a="pos")
            except np.linalg.LinAlgError:
                K[t] = np.linalg.pinv(M1) @ M2
            Acl = A - B @ K[t]
            S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            S[t] = (S[t] + S[t].T) / 2

        U = np.zeros((T, m))
        x = x0
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.ravel()
            x = A @ x + B @ u

        # Debug
        J_opt = float(x0.T @ S[0] @ x0)
        print(f"DEBUG: J_opt={J_opt}")
        
        J = 0.0
        x_sim = x0
        for t in range(T):
            u_sim = U[t].reshape(-1, 1)
            J += float(x_sim.T @ Q @ x_sim + u_sim.T @ R @ u_sim)
            x_sim = A @ x_sim + B @ u_sim
        J += float(x_sim.T @ P @ x_sim)
        print(f"DEBUG: J={J}")

        return {"U": U}