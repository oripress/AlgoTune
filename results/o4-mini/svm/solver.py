from typing import Any, Dict
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solves the SVM using CVXPY and returns
            beta0 : float
            beta  : list[float]
            optimal_value : float
            missclass_error : float
        """
        # Load data
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)[:, None]
        C = float(problem["C"])

        # Problem dimensions
        p = X.shape[1]
        n = X.shape[0]

        # CVXPY variables
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        # Objective and constraints
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]

        # Form and solve problem
        prob = cp.Problem(objective, constraints)
        try:
            optimal_value = prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        # Check status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        if beta.value is None or beta0.value is None:
            return None

        # Compute predictions and misclassification
        pred = X @ beta.value + beta0.value
        missclass = np.mean((pred * y) < 0)

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }