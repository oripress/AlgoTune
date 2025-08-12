import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, hstack, csr_matrix
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit a quantile regression model using a sparse linear‑programming
        formulation (HiGHS solver) and return coefficients, intercept,
        and in‑sample predictions.
        """
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        tau = float(problem["quantile"])
        fit_intercept = bool(problem["fit_intercept"])

        n_samples, n_features = X.shape

        # Augment with intercept column if needed
        if fit_intercept:
            X_aug = np.hstack([X, np.ones((n_samples, 1))])
        else:
            X_aug = X

        p = X_aug.shape[1]                     # number of beta variables

        # Objective: minimize tau*sum(u) + (1-tau)*sum(v)
        c = np.concatenate([
            np.zeros(p),                       # beta coefficients have zero cost
            tau * np.ones(n_samples),          # u >= 0
            (1.0 - tau) * np.ones(n_samples)  # v >= 0
        ])

        # Equality constraints: X_aug * beta + u - v = y
        # Build a sparse A_eq matrix to avoid O(n²) memory/time overhead
        A_eq = hstack([
            csr_matrix(X_aug),   # (n, p)
            eye(n_samples, format='csr'),   # u coefficients (+1)
            -eye(n_samples, format='csr')   # v coefficients (-1)
        ], format='csr')
        b_eq = y

        # Bounds: beta unrestricted, u >= 0, v >= 0
        bounds = [(None, None)] * p + [(0, None)] * (2 * n_samples)

        # Solve with HiGHS (fast interior‑point)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if not res.success:
            raise RuntimeError("Linear programming failed to converge.")

        beta = res.x[:p]

        if fit_intercept:
            coef = beta[:-1].tolist()
            intercept = [beta[-1]]
        else:
            coef = beta.tolist()
            intercept = [0.0]

        predictions = (X @ np.array(coef) + intercept[0]).tolist()

        return {"coef": coef, "intercept": intercept, "predictions": predictions}