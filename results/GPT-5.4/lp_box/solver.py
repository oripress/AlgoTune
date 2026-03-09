from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[float]]:
        c = np.asarray(problem["c"], dtype=np.float64)
        n = c.size
        if n == 0:
            return {"solution": []}

        b = np.asarray(problem["b"], dtype=np.float64)
        if b.size == 0:
            return {"solution": (c < 0.0).astype(np.float64).tolist()}

        A = np.asarray(problem["A"], dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, n)

        if np.all(c >= 0.0) and np.all(b >= -1e-12):
            return {"solution": np.zeros(n, dtype=np.float64).tolist()}

        x = (c < 0.0).astype(np.float64)
        if np.all(A @ x <= b + 1e-12):
            return {"solution": x.tolist()}

        if np.all(c <= 0.0):
            x1 = np.ones(n, dtype=np.float64)
            if np.all(A @ x1 <= b + 1e-12):
                return {"solution": x1.tolist()}

        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=(0.0, 1.0),
            method="highs",
        )
        if not res.success:
            res = linprog(
                c,
                A_ub=A,
                b_ub=b,
                bounds=(0.0, 1.0),
                method="highs-ds",
            )
        if not res.success:
            res = linprog(
                c,
                A_ub=A,
                b_ub=b,
                bounds=(0.0, 1.0),
                method="highs-ipm",
            )

        return {"solution": np.asarray(res.x, dtype=np.float64).tolist()}