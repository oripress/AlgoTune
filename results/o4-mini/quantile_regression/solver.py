import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sparse

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        tau = float(problem["quantile"])
        fit_intercept = bool(problem["fit_intercept"])
        n, p = X.shape

        # Add intercept column if requested
        if fit_intercept:
            ones = np.ones((n, 1), dtype=float)
            X_ = np.hstack((ones, X))
            p_ = p + 1
        else:
            X_ = X
            p_ = p

        # Objective: minimize tau * sum(u) + (1 - tau) * sum(v)
        # Variables: [beta (length p_), u (n), v (n)]
        c = np.concatenate([
            np.zeros(p_, dtype=float),
            tau * np.ones(n, dtype=float),
            (1 - tau) * np.ones(n, dtype=float),
        ])

        # Bounds: beta free, u >= 0, v >= 0
        bounds = [(None, None)] * p_ + [(0, None)] * (2 * n)

        # Equality constraints: X_*beta + u - v = y
        A_beta = sparse.csr_matrix(X_)
        I_n = sparse.identity(n, format='csr')
        A_eq = sparse.hstack([A_beta, I_n, -I_n], format='csr')
        b_eq = y

        # Solve LP with HiGHS
        res = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        if not res.success:
            raise RuntimeError(f"LP failure in quantile regression: {res.message}")

        sol = res.x
        beta = sol[:p_]

        # Extract intercept and coefficients
        if fit_intercept:
            intercept = float(beta[0])
            coef = beta[1:]
        else:
            intercept = 0.0
            coef = beta

        # Compute in-sample predictions
        preds = X.dot(coef) + intercept

        return {
            "coef": coef.tolist(),
            "intercept": [intercept],
            "predictions": preds.tolist(),
        }