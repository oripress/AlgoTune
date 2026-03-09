from typing import Any

import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        try:
            X = np.asarray(problem["X"], dtype=float)
            y = np.asarray(problem["y"], dtype=float).reshape(-1, 1)
            C = float(problem["C"])
        except Exception:
            return None

        if X.ndim != 2 or y.ndim != 2 or y.shape[1] != 1 or X.shape[0] != y.shape[0] or C <= 0:
            return None

        n, p = X.shape
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]
        prob = cp.Problem(objective, constraints)

        try:
            optimal_value = prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            return None
        except Exception:
            return None

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        if beta.value is None or beta0.value is None:
            return None

        pred = X @ beta.value + beta0.value
        missclass = np.mean((pred * y) < 0)

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.reshape(-1).tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }