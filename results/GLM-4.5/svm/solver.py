from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solves the SVM using CVXPY with optimized formulation.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)[:, None]
        C = float(problem["C"])

        p, n = X.shape[1], X.shape[0]
        
        # Pre-compute y*X for more efficient constraint formulation
        yX = y * X

        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            yX @ beta + y * beta0 >= 1 - xi,
        ]

        prob = cp.Problem(objective, constraints)
        
        # Use default solver
        optimal_value = prob.solve(verbose=False)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Use more efficient computation for prediction
        beta_val = beta.value
        beta0_val = beta0.value
        pred = X @ beta_val + beta0_val
        missclass = np.mean((pred * y) < 0)

        return {
            "beta0": float(beta0_val),
            "beta": beta_val.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }