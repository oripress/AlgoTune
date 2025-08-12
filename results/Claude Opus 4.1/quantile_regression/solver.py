import numpy as np
from scipy.optimize import linprog
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Optimized quantile regression using direct LP formulation with scipy.
        """
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        quantile = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        
        # Add intercept column if needed
        if fit_intercept:
            X_aug = np.column_stack([np.ones(n_samples), X])
            n_params = n_features + 1
        else:
            X_aug = X
            n_params = n_features
        
        # Linear programming formulation of quantile regression
        # Variables: [beta, u, v] where u and v are positive/negative residuals
        # Minimize: quantile * sum(u) + (1-quantile) * sum(v)
        # Subject to: X*beta + u - v = y, u >= 0, v >= 0
        
        # Objective coefficients
        c = np.zeros(n_params + 2 * n_samples)
        c[n_params:n_params + n_samples] = quantile
        c[n_params + n_samples:] = 1 - quantile
        
        # Equality constraints: X*beta + u - v = y
        A_eq = np.zeros((n_samples, n_params + 2 * n_samples))
        A_eq[:, :n_params] = X_aug
        A_eq[:, n_params:n_params + n_samples] = np.eye(n_samples)
        A_eq[:, n_params + n_samples:] = -np.eye(n_samples)
        b_eq = y
        
        # Bounds: beta unbounded, u >= 0, v >= 0
        bounds = [(None, None)] * n_params + [(0, None)] * (2 * n_samples)
        
        # Solve
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, 
                        method='highs-ds', options={'disp': False})
        
        # Extract beta
        beta_val = result.x[:n_params]
        
        if fit_intercept:
            intercept = [float(beta_val[0])]
            coef = beta_val[1:].tolist()
        else:
            intercept = [0.0]
            coef = beta_val.tolist()
        
        # Compute predictions
        predictions = (X_aug @ beta_val).tolist()
        
        return {
            "coef": coef,
            "intercept": intercept,
            "predictions": predictions
        }