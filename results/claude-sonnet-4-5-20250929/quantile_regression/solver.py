import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit quantile regression using scipy's linprog with HiGHS and sparse matrices.
        
        :param problem: dict with 'X', 'y', 'quantile', 'fit_intercept'
        :return: dict with 'coef', 'intercept', 'predictions'
        """
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        tau = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        
        # Prepare design matrix
        if fit_intercept:
            X_design = np.column_stack([np.ones(n_samples), X])
            n_params = n_features + 1
        else:
            X_design = X
            n_params = n_features
        
        # Formulate as LP: min tau*sum(u) + (1-tau)*sum(v)
        # subject to: X*beta + u - v = y
        # Variables: [beta, u, v]
        
        # Objective: c = [0...0, tau...tau, (1-tau)...(1-tau)]
        c = np.concatenate([
            np.zeros(n_params),
            np.full(n_samples, tau),
            np.full(n_samples, 1 - tau)
        ])
        
        # Equality constraint: [X, I, -I] * [beta, u, v]^T = y
        # Use sparse matrices for efficiency
        X_sparse = csr_matrix(X_design)
        I_sparse = csr_matrix(np.eye(n_samples))
        A_eq = sparse_hstack([X_sparse, I_sparse, -I_sparse])
        b_eq = y
        
        # Solve
        result = linprog(
            c, A_eq=A_eq, b_eq=b_eq,
            bounds=[(None, None)] * n_params + [(0, None)] * (2 * n_samples),
            method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-7, 'primal_feasibility_tolerance': 1e-7}
        )
        
        beta_val = result.x[:n_params]
        
        if fit_intercept:
            intercept = [float(beta_val[0])]
            coef = beta_val[1:].tolist()
            predictions = (X @ beta_val[1:] + beta_val[0]).tolist()
        else:
            intercept = [0.0]
            coef = beta_val.tolist()
            predictions = (X @ beta_val).tolist()
        
        return {"coef": coef, "intercept": intercept, "predictions": predictions}