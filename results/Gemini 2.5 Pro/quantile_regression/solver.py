import numpy as np
from scipy.optimize import linprog
from scipy.sparse import hstack, eye
from typing import Any

class Solver:
    def _solve_primal_sparse(self, X_aug, y, quantile, p):
        """
        Solves the primal LP using scipy.linprog with sparse constraint matrices.
        """
        n_samples = X_aug.shape[0]
        
        c = np.concatenate([np.zeros(p),
                            np.full(n_samples, quantile),
                            np.full(n_samples, 1 - quantile)])

        A_eq = hstack([X_aug, eye(n_samples), -eye(n_samples)], format='csr')
        b_eq = y

        bounds = [(None, None)] * p + [(0, None)] * (2 * n_samples)

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                      options={'presolve': True})
        
        return res.x[:p] if res.success else np.zeros(p)

    def _solve_dual(self, X_aug, y, quantile, p):
        """
        Solves the dual LP using scipy.linprog.
        """
        n_samples = X_aug.shape[0]
        c = -y
        A_eq = X_aug.T
        b_eq = np.zeros(p)
        bounds = [(-(1 - quantile), quantile)] * n_samples

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                      options={'presolve': True})

        if res.success:
            return -res.eqlin.marginals
        else:
            return self._solve_primal_sparse(X_aug, y, quantile, p)

    def solve(self, problem: dict[str, Any]) -> Any:
        """
        Quantile regression solver using scipy.optimize.linprog.
        It automatically chooses between the primal and dual LP formulation
        based on the dimensions of the problem.
        The primal formulation uses sparse matrices for efficiency.
        """
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        quantile = problem["quantile"]
        fit_intercept = problem["fit_intercept"]

        n_samples, n_features = X.shape

        if fit_intercept:
            X_aug = np.c_[X, np.ones(n_samples, dtype=np.float64)]
            p = n_features + 1
        else:
            X_aug = X
            p = n_features

        if p == 0:
            beta = np.array([])
        elif n_samples > p:
            beta = self._solve_dual(X_aug, y, quantile, p)
        else:
            beta = self._solve_primal_sparse(X_aug, y, quantile, p)

        if fit_intercept:
            if p > 0:
                coef = beta[:-1]
                intercept = beta[-1]
            else:
                coef = np.array([])
                intercept = 0.0
        else:
            coef = beta
            intercept = 0.0
        
        predictions = X_aug @ beta if p > 0 else np.zeros(n_samples)

        return {
            "coef": coef.tolist(),
            "intercept": [intercept],
            "predictions": predictions.tolist()
        }