# coding: utf-8
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the projection onto the CVaR constraint set using CVXPY.
        """
        # Extract data
        x0 = np.array(problem["x0"], dtype=float)
        A = np.array(problem["loss_scenarios"], dtype=float)
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 0.0))
        m, n = A.shape

        # Number of worst-case scenarios
        k = int((1.0 - beta) * m)
        if k <= 0:
            # No tail, already feasible
            return {"x_proj": x0.tolist()}

        # Define CVXPY problem
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        # CVaR constraint: sum of k largest losses ≤ kappa * k
        constraint = [cp.sum_largest(A @ x, k) <= kappa * k]

        prob = cp.Problem(objective, constraint)
        prob.solve(solver=cp.OSQP)

        # Check solution
        xv = x.value
        if xv is None:
            return {"x_proj": []}
        return {"x_proj": xv.tolist()}