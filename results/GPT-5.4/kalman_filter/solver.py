from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    __slots__ = ("_empty",)

    def __init__(self) -> None:
        self._empty = {"x_hat": [], "w_hat": [], "v_hat": []}

    def solve(self, problem, **kwargs) -> Any:
        try:
            A = np.asarray(problem["A"], dtype=np.float64)
            B = np.asarray(problem["B"], dtype=np.float64)
            C = np.asarray(problem["C"], dtype=np.float64)
            y = np.asarray(problem["y"], dtype=np.float64)
            x0 = np.asarray(problem["x_initial"], dtype=np.float64)
            tau = float(problem["tau"])
        except Exception:
            return self._empty

        try:
            if tau <= 0.0:
                return self._empty
            if A.ndim != 2 or B.ndim != 2 or C.ndim != 2 or y.ndim != 2 or x0.ndim != 1:
                return self._empty

            N, m = y.shape
            n = A.shape[1]

            if A.shape != (n, n):
                return self._empty
            if B.shape[0] != n or C.shape[1] != n or C.shape[0] != m or x0.shape[0] != n:
                return self._empty

            x_hat, w_hat, v_hat = self._solve_riccati(A, B, C, y, x0, tau)
            return {"x_hat": x_hat, "w_hat": w_hat, "v_hat": v_hat}
        except Exception:
            return self._empty

    @staticmethod
    @staticmethod
    def _solve_riccati(A, B, C, y, x0, tau):
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        CT = C.T

        x_hat = np.empty((N + 1, n), dtype=np.float64)
        x_hat[0] = x0

        if N == 0:
            return x_hat, np.empty((0, p), dtype=np.float64), np.empty((0, m), dtype=np.float64)

        if p == 0:
            w_hat = np.empty((N, 0), dtype=np.float64)
            for t in range(N):
                x_hat[t + 1] = A @ x_hat[t]
            v_hat = y - x_hat[:-1] @ CT
            return x_hat, w_hat, v_hat

        Q = tau * (CT @ C)
        q = tau * (y @ C)
        AT = A.T
        BT = B.T

        P = np.zeros((n, n), dtype=np.float64)
        r = np.zeros(n, dtype=np.float64)

        K_seq = np.empty((N, p, n), dtype=np.float64)
        d_seq = np.empty((N, p), dtype=np.float64)

        if p <= n:
            RHS = np.empty((p, n + 1), dtype=np.float64)
            for t in range(N - 1, -1, -1):
                PB = P @ B
                PA = P @ A

                M = BT @ PB
                M.flat[:: p + 1] += 1.0
                RHS[:, :n] = BT @ PA
                RHS[:, n] = BT @ r
                sol = np.linalg.solve(M, RHS)

                K = sol[:, :n]
                d = sol[:, n]

                K_seq[t] = K
                d_seq[t] = d

                P = Q + AT @ (PA - PB @ K)
                r = q[t] + AT @ (r - PB @ d)
        else:
            RHS = np.empty((n, n + 1), dtype=np.float64)
            for t in range(N - 1, -1, -1):
                PB = P @ B
                PA = P @ A

                M = PB @ BT
                M.flat[:: n + 1] += 1.0
                RHS[:, :n] = PA
                RHS[:, n] = r
                sol = np.linalg.solve(M, RHS)

                U = sol[:, :n]
                u = sol[:, n]
                K = BT @ U
                d = BT @ u

                K_seq[t] = K
                d_seq[t] = d

                P = Q + AT @ (PA - PB @ K)
                r = q[t] + AT @ (r - PB @ d)

        w_hat = np.empty((N, p), dtype=np.float64)
        for t in range(N):
            xt = x_hat[t]
            wt = d_seq[t] - K_seq[t] @ xt
            w_hat[t] = wt
            x_hat[t + 1] = A @ xt + B @ wt

        v_hat = y - x_hat[:-1] @ CT
        return x_hat, w_hat, v_hat