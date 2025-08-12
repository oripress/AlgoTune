from typing import Any, Dict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fit quantile regression via a linear program using SciPy's HiGHS solver.
        This mirrors scikit-learn's QuantileRegressor with alpha=0 and solver='highs',
        but with a lightweight, vectorized LP assembly for speed.
        """
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        tau = float(problem["quantile"])
        fit_intercept = bool(problem["fit_intercept"])

        n, p = X.shape if X.ndim == 2 else (len(X), 0)

        # Variables: [beta (p), intercept (1 if fit), u (n), v (n)]
        n_beta = p
        n_b = 1 if fit_intercept else 0
        n_u = n
        n_v = n
        n_vars = n_beta + n_b + n_u + n_v

        # Objective: minimize tau * sum(u) + (1 - tau) * sum(v)
        c = np.zeros(n_vars, dtype=float)
        c[n_beta + n_b : n_beta + n_b + n_u] = tau
        c[n_beta + n_b + n_u :] = 1.0 - tau

        # Equality constraints: X * beta + b + u - v = y
        rows_parts = []
        cols_parts = []
        data_parts = []

        if n_beta > 0:
            rr, cc = np.nonzero(X)
            if rr.size > 0:
                rows_parts.append(rr.astype(np.int64, copy=False))
                cols_parts.append(cc.astype(np.int64, copy=False))
                data_parts.append(X[rr, cc].astype(float, copy=False))

        if fit_intercept:
            rows_b = np.arange(n, dtype=np.int64)
            cols_b = np.full(n, n_beta, dtype=np.int64)
            data_b = np.ones(n, dtype=float)
            rows_parts.append(rows_b)
            cols_parts.append(cols_b)
            data_parts.append(data_b)

        if n_u:
            rows_u = np.arange(n, dtype=np.int64)
            cols_u = n_beta + n_b + rows_u
            data_u = np.ones(n, dtype=float)
            rows_parts.append(rows_u)
            cols_parts.append(cols_u)
            data_parts.append(data_u)

        if n_v:
            rows_v = np.arange(n, dtype=np.int64)
            cols_v = n_beta + n_b + n_u + rows_v
            data_v = -np.ones(n, dtype=float)
            rows_parts.append(rows_v)
            cols_parts.append(cols_v)
            data_parts.append(data_v)

        if rows_parts:
            rows = np.concatenate(rows_parts)
            cols = np.concatenate(cols_parts)
            data = np.concatenate(data_parts)
        else:
            rows = np.empty(0, dtype=np.int64)
            cols = np.empty(0, dtype=np.int64)
            data = np.empty(0, dtype=float)

        A_eq = coo_matrix((data, (rows, cols)), shape=(n, n_vars))
        b_eq = y

        # Bounds: beta, b are free; u >= 0; v >= 0
        bounds = []
        if n_beta > 0:
            bounds.extend([(None, None)] * n_beta)
        if fit_intercept:
            bounds.append((None, None))
        bounds.extend([(0.0, None)] * n_u)
        bounds.extend([(0.0, None)] * n_v)

        res = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if res.status != 0 or res.x is None:
            # Robust fallback (rare): neutral solution
            beta = np.zeros(p, dtype=float)
            b_val = float(np.quantile(y, tau)) if fit_intercept else 0.0
        else:
            x = res.x
            beta = x[:n_beta] if n_beta > 0 else np.zeros(p, dtype=float)
            b_val = x[n_beta] if fit_intercept else 0.0

        intercept = [float(b_val)]
        preds = (X @ beta + (b_val if fit_intercept else 0.0)) if p > 0 else np.full(n, b_val, dtype=float)

        return {
            "coef": beta.tolist(),
            "intercept": intercept,
            "predictions": preds.tolist(),
        }