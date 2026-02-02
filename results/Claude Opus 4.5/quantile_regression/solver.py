import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        quantile = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        
        beta = cp.Variable(n_features)
        if fit_intercept:
            intercept = cp.Variable()
            pred = X @ beta + intercept
        else:
            pred = X @ beta
        
        residuals = y - pred
        
        # Quantile loss using pinball loss formulation
        u = cp.Variable(n_samples, nonneg=True)
        v = cp.Variable(n_samples, nonneg=True)
        
        constraints = [residuals == u - v]
        objective = cp.Minimize(quantile * cp.sum(u) + (1 - quantile) * cp.sum(v))
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        beta_val = beta.value
        intercept_val = intercept.value if fit_intercept else 0.0
        
        predictions = X @ beta_val + intercept_val
        
        return {
            "coef": beta_val.tolist(),
            "intercept": [float(intercept_val)],
            "predictions": predictions.tolist()
        }