import numpy as np
from scipy.optimize import linprog
from scipy import sparse
from sklearn.linear_model import QuantileRegressor

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        tau = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        
        # Add intercept column if needed
        if fit_intercept:
            X = np.hstack([X, np.ones((n_samples, 1))])
            n_features += 1
        
        # Problem dimensions
        n_vars = n_features + 2 * n_samples  # β, u, v
        
        # Cost vector: [0 for β, tau for u, 1-tau for v]
        c = np.zeros(n_vars)
        c[n_features:n_features+n_samples] = tau          # u costs
        c[n_features+n_samples:] = 1 - tau                # v costs
        
        # Equality constraints: Xβ + u - v = y
        A_eq = sparse.hstack([
            sparse.csr_matrix(X),                         # β coefficients
            sparse.eye(n_samples),                        # u coefficients
            -sparse.eye(n_samples)                        # v coefficients
        ])
        
        # Bounds: 
        #   β: unbounded
        #   u: >= 0
        #   v: >= 0
        bounds = [(None, None)] * n_features + [(0, None)] * (2 * n_samples)
        
        # Solve linear program
        res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, 
                      method='highs', options={'disp': False})
        
        if res.success:
            # Extract solution
            beta = res.x[:n_features]
            
            # Split coefficients and intercept
            if fit_intercept:
                coef = beta[:-1].tolist()
                intercept = [beta[-1]]
            else:
                coef = beta.tolist()
                intercept = [0.0]
            
            # Calculate predictions
            predictions = X.dot(beta).tolist()
        else:
            # Use reference implementation as fallback
            model = QuantileRegressor(
                quantile=tau,
                alpha=0.0,
                fit_intercept=fit_intercept,
                solver="highs",
            )
            model.fit(X[:, :-1] if fit_intercept else X, y)
            coef = model.coef_.tolist()
            intercept = [model.intercept_]
            predictions = model.predict(X[:, :-1] if fit_intercept else X).tolist()
        
        return {
            "coef": coef,
            "intercept": intercept,
            "predictions": predictions
        }