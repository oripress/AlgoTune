from typing import Any, Dict

import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solves the SVM using CVXPY and returns:
            beta0 : float
            beta  : list[float]
            optimal_value : float
            missclass_error : float

        The formulation matches the reference implementation to ensure
        consistent numerical results.
        """
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float).reshape(-1, 1)
        C = float(problem["C"])

        p, n = X.shape[1], X.shape[0]

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
            optimal_value = prob.solve()
        except Exception:
            # Fallback attempt if default solver throws
            try:
                optimal_value = prob.solve(solver=cp.OSQP)
            except Exception:
                return None

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # As a last resort, try ECOS
            try:
                optimal_value = prob.solve(solver=cp.ECOS)
            except Exception:
                return None

        if beta.value is None or beta0.value is None or not np.isfinite(optimal_value):
            return None

        pred = X @ beta.value + beta0.value
        missclass = np.mean((pred * y) < 0)

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }