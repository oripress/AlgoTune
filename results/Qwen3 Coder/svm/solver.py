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
        # Extract problem data
        X = np.array(problem["X"])
        y = np.array(problem["y"])[:, None]  # Reshape like reference
        C = float(problem["C"])

        n, p = X.shape

        # Define variables
        beta = cp.Variable((p, 1))  # Match reference shape
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))    # Match reference shape

        # Define objective function - use explicit expressions for better performance
        # Define objective and constraints in a more efficient way
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))

        # Define constraints (matching reference exactly)
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]

        # Create and solve the problem with optimized settings
        prob = cp.Problem(objective, constraints)
        try:
            optimal_value = prob.solve(verbose=False, warm_start=True)
        except Exception:
            return None

        # Check if solution is valid
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Calculate misclassification error
        pred = X @ beta.value + beta0.value
        missclass = np.mean((pred * y) < 0)

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),  # Flatten to match expected output
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }