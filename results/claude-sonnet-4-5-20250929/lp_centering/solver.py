from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp centering problem using CVXPY.
        
        :param problem: A dictionary of the lp centering problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp centering problem.
        """
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        n = c.shape[0]

        x = cp.Variable(n)
        
        # Try OSQP solver first (often faster for smaller problems)
        prob = cp.Problem(cp.Minimize(c.T @ x - cp.sum(cp.log(x))), [A @ x == b])
        try:
            prob.solve(solver="OSQP", eps_abs=1e-7, eps_rel=1e-7)
            if prob.status == "optimal":
                return {"solution": x.value.tolist()}
        except:
            pass
        
        # Fallback to CLARABEL
        prob.solve(solver="CLARABEL")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}