from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solves the soft-margin SVM:
            min_{beta,beta0,xi} 0.5||beta||^2 + C * sum(xi)
            s.t. xi >= 0
                 y_i * (x_i^T beta + beta0) >= 1 - xi_i

        Returns a dict with keys:
          - "beta0": float
          - "beta": list[float]
          - "optimal_value": float
          - "missclass_error": float
        """
        # Parse inputs
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)[:, None]
        C = float(problem["C"])

        n, p = X.shape

        # Variables
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        # Objective and constraints
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]

        prob = cp.Problem(objective, constraints)

        try:
            optimal_value = prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Misclassification error
        pred = X @ beta.value + beta0.value  # shape (n,1)
        missclass = float(np.mean((pred * y) < 0))

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass,
        }