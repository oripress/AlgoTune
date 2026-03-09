from typing import Any

import numpy as np
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        a, b = problem["A"], problem["B"]
        q, r, p = problem["Q"], problem["R"], problem["P"]
        horizon, x0 = problem["T"], problem["x0"]

        n, m = b.shape
        s = np.zeros((horizon + 1, n, n))
        gains = np.zeros((horizon, m, n))
        s[horizon] = p

        for t in range(horizon - 1, -1, -1):
            st1 = s[t + 1]
            m1 = r + b.T @ st1 @ b
            m2 = b.T @ st1 @ a
            try:
                gains[t] = linalg_solve(m1, m2, assume_a="pos")
            except np.linalg.LinAlgError:
                gains[t] = np.linalg.pinv(m1) @ m2
            acl = a - b @ gains[t]
            s[t] = q + gains[t].T @ r @ gains[t] + acl.T @ st1 @ acl
            s[t] = (s[t] + s[t].T) / 2

        u = np.zeros((horizon, m))
        x = x0
        for t in range(horizon):
            ut = -gains[t] @ x
            u[t] = ut.ravel()
            x = a @ x + b @ ut

        return {"U": u}