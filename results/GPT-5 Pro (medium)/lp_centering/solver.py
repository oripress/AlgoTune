from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Solve the LP centering problem using CVXPY with the Clarabel solver,
        matching the reference implementation for numerical consistency.

            minimize    c^T x - sum_i log(x_i)
            subject to  A x = b

        Returns a dict with key "solution" mapping to the optimal x as a list.
        """
        c = np.array(problem["c"], dtype=float)
        A = np.array(problem["A"], dtype=float)
        b = np.array(problem["b"], dtype=float)
        n = c.shape[0]

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c.T @ x - cp.sum(cp.log(x))), [A @ x == b])
        prob.solve(solver="CLARABEL")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}