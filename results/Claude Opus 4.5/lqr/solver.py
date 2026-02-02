from typing import Any
import numpy as np
import numpy.testing as npt
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        Q = np.array(problem["Q"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        P = np.array(problem["P"], dtype=np.float64)
        T = int(problem["T"])
        x0 = np.array(problem["x0"], dtype=np.float64).reshape(-1, 1)

        n, m = B.shape
        S = np.zeros((T + 1, n, n), dtype=np.float64)
        K = np.zeros((T, m, n), dtype=np.float64)
        S[T] = P.copy()

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
            S[t] = (S[t] + S[t].T) / 2.0

        U = np.zeros((T, m), dtype=np.float64)
        x = x0.copy()
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.ravel()
            x = A @ x + B @ u

        return {"U": U}