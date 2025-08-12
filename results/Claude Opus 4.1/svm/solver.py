from typing import Any, Dict
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solves the SVM using CVXPY with optimizations.
        """
        X = np.array(problem["X"])
        y = np.array(problem["y"])[:, None]
        C = float(problem["C"])
        
        p, n = X.shape[1], X.shape[0]
        
        # Variables
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))
        
        # Objective
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        
        # Constraints
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]
        
        # Create and solve problem
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
        
        # Calculate misclassification error
        pred = X @ beta.value + beta0.value
        missclass = np.mean((pred * y) < 0)
        
        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }