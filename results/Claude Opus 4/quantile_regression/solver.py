import numpy as np
from scipy.optimize import linprog
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit quantile regression using scipy's linprog for better performance.
        
        :param problem: dict with 'X', 'y', 'quantile', 'fit_intercept'
        :return: dict with 'coef', 'intercept', 'predictions'
        """
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        quantile = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        
        # Variables: [beta, intercept (if needed), u, v]
        # where u and v are slack variables for positive and negative residuals
        if fit_intercept:
            n_vars = n_features + 1 + 2 * n_samples
            # Construct extended X matrix with intercept column
            X_ext = np.column_stack([X, np.ones(n_samples)])
            n_coef = n_features + 1
        else:
            n_vars = n_features + 2 * n_samples
            X_ext = X
            n_coef = n_features
        
        # Objective: minimize sum(quantile * u + (1-quantile) * v)
        c = np.zeros(n_vars)
        c[n_coef:n_coef + n_samples] = quantile  # u coefficients
        c[n_coef + n_samples:] = 1 - quantile    # v coefficients
        
        # Equality constraints: y - X*beta - intercept = u - v
        # Rearranged: X*beta + intercept + u - v = y
        A_eq = np.zeros((n_samples, n_vars))
        A_eq[:, :n_coef] = X_ext
        A_eq[:, n_coef:n_coef + n_samples] = np.eye(n_samples)  # u
        A_eq[:, n_coef + n_samples:] = -np.eye(n_samples)       # v
        b_eq = y
        
        # Bounds: beta unbounded, u >= 0, v >= 0
        bounds = [(None, None)] * n_coef + [(0, None)] * (2 * n_samples)
        
        # Solve
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        # Extract results
        if fit_intercept:
            coef = result.x[:n_features].tolist()
            intercept_val = [result.x[n_features]]
        else:
            coef = result.x[:n_features].tolist()
            intercept_val = [0.0]
        
        # Compute predictions
        predictions = (X @ np.array(coef) + intercept_val[0]).tolist()
        
        return {"coef": coef, "intercept": intercept_val, "predictions": predictions}