from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, Any]:
        """
        Solves a given robust LP using CVXPY with ECOS solver.
        """
        c = np.array(problem["c"])
        b = np.array(problem["b"])
        P = np.array(problem["P"])
        q = np.array(problem["q"])
        m = len(P)
        n = len(c)

        x = cp.Variable(n)

        constraints = []
        for i in range(m):
            constraints.append(cp.SOC(b[i] - q[i].T @ x, P[i].T @ x))

        cvx_problem = cp.Problem(cp.Minimize(c.T @ x), constraints)

        try:
            # Try ECOS which is typically fast for SOCP
            cvx_problem.solve(solver=cp.ECOS, verbose=False)

            # Check if a solution was found
            if cvx_problem.status not in ["optimal", "optimal_inaccurate"]:
                return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}

            return {"objective_value": cvx_problem.value, "x": x.value}

        except Exception:
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}