import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)[:, None]
        C = float(problem["C"])

        # Problem dimensions
        n, p = X.shape[0], X.shape[1]

        # Define variables
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        # Objective and constraints
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]

        # Solve the QP
        prob = cp.Problem(objective, constraints)
        optimal_value = prob.solve()

        # Extract solution
        beta_val = beta.value
        beta0_val = beta0.value
        # Compute predictions and misclassification
        decision = X @ beta_val + beta0_val
        missclass = np.mean((decision * y) < 0)

        return {
            "beta0": float(beta0_val),
            "beta": beta_val.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }