from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        X = np.array(problem["X"])
        y = np.array(problem["y"])[:, None]
        C = float(problem["C"])

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
            optimal_value = prob.solve()
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
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }