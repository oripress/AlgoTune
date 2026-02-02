from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, list]:
        import cvxpy as cp

        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        n = int(c.size)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c.T @ x - cp.sum(cp.log(x))), [A @ x == b])
        prob.solve(solver="CLARABEL", warm_start=True)
        return {"solution": x.value.tolist()}