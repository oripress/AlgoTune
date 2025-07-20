import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Reference CVXPY solver for LP centering:
          minimize   c^T x - sum(log x)
          subject to A x = b, x > 0.
        """
        c = np.array(problem["c"], dtype=float)
        A = np.array(problem["A"], dtype=float)
        b = np.array(problem["b"], dtype=float)
        n = c.shape[0]

        # Define variable and problem
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(c.T @ x - cp.sum(cp.log(x))),
            [A @ x == b]
        )
        # Solve with the same solver as reference
        prob.solve(solver="CLARABEL")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}