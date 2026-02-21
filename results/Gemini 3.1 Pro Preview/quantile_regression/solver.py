import numpy as np
from scipy.optimize import linprog
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        quantile = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        
        c = -y
        
        if fit_intercept:
            A_eq = np.vstack([X.T, np.ones(n_samples)])
            b_eq = np.concatenate([(1 - quantile) * X.T @ np.ones(n_samples), [(1 - quantile) * n_samples]])
        else:
            A_eq = X.T
            b_eq = (1 - quantile) * X.T @ np.ones(n_samples)
            
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
        
        marginals = -res.eqlin.marginals
        
        if fit_intercept:
            coef = marginals[:-1]
            intercept = marginals[-1]
        else:
            coef = marginals
            intercept = 0.0
            
        coef = coef.tolist()
        intercept = [float(intercept)]
        
        # Calculate predictions
        predictions = (X @ np.array(coef) + intercept[0]).tolist()
        
        return {
            "coef": coef,
            "intercept": intercept,
            "predictions": predictions
        }