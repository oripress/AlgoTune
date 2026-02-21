import numpy as np
from numba import njit
from typing import Any

@njit(cache=True, fastmath=True, boundscheck=False)
def _solve_numba(A, B, C, y, x0, tau):
    N, m = y.shape
    n = A.shape[1]
    p = B.shape[1]

    BBT = B @ B.T
    tauCTC = tau * (C.T @ C)
    tauCTy = tau * (y @ C)

    P = np.zeros((N + 1, n, n))
    q = np.zeros((N + 1, n))
    F = np.zeros((N + 1, n, n))

    I = np.eye(n)
    converged = False
    AT_F = np.zeros((n, n))
    for t in range(N - 1, -1, -1):
        if not converged:
            F[t + 1] = np.linalg.inv(I + P[t + 1] @ BBT)
            AT_F = A.T @ F[t + 1]
            P[t] = AT_F @ P[t + 1] @ A + tauCTC
            
            diff = 0.0
            for i in range(n):
                for j in range(n):
                    d = abs(P[t, i, j] - P[t + 1, i, j])
                    if d > diff:
                        diff = d
            if diff < 1e-9:
                converged = True
        else:
            F[t + 1] = F[t + 2]
            P[t] = P[t + 1]
            
        q[t] = AT_F @ q[t + 1] + tauCTy[t]
    x = np.zeros((N + 1, n))
    w = np.zeros((N, p))
    v = np.zeros((N, m))

    x[0] = x0
    for t in range(N):
        x[t + 1] = F[t + 1].T @ (A @ x[t] + BBT @ q[t + 1])
        w[t] = -B.T @ (P[t + 1] @ x[t + 1] - q[t + 1])
        v[t] = y[t] - C @ x[t]

    return x, w, v

class Solver:
    def __init__(self):
        # Warm up numba compilation
        A = np.eye(2)
        B = np.eye(2)
        C = np.eye(2)
        y = np.zeros((2, 2))
        x0 = np.zeros(2)
        tau = 1.0
        _solve_numba(A, B, C, y, x0, tau)

    def solve(self, problem: dict, **kwargs) -> Any:
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        C = np.array(problem["C"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        x0 = np.array(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])

        x, w, v = _solve_numba(A, B, C, y, x0, tau)

        return {
            "x_hat": x.tolist(),
            "w_hat": w.tolist(),
            "v_hat": v.tolist(),
        }