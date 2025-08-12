import cvxpy as cp
import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the LP centering problem:
            maximize    cᵀx - Σ log(x_i)
            subject to  A x = b,   x > 0
        """
        c = np.array(problem["c"], dtype=float)
        A = np.array(problem["A"], dtype=float)
        b = np.array(problem["b"], dtype=float)

        n = c.shape[0]

        # Decision variable
        x = cp.Variable(n)

        # Small lower bound to keep log defined
        eps = 1e-6
        constraints = [A @ x == b, x >= eps]

        # Convex objective (minimization) as per DCP rules
        objective = cp.Minimize(c @ x - cp.sum(cp.log(x)))
        # Define the problem
        prob = cp.Problem(objective, constraints)
        # Solve using ECOS solver (supports exponential cone)
        prob.solve(solver=cp.ECOS, verbose=False)

        if x.value is None:
            raise RuntimeError("Failed to obtain a solution for the LP centering problem.")
        return {"solution": x.value.tolist()}