from typing import Any, Dict
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Quantile regression via linear programming.

        Minimize sum_i [ tau * u_i + (1 - tau) * v_i ]
        subject to X beta + intercept + u - v = y
                  u >= 0, v >= 0

        Variable ordering: [beta (p), intercept (optional), u (n), v (n)]

        Returns dict with keys "coef", "intercept", "predictions".
        """
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        tau = float(problem["quantile"])
        fit_intercept = bool(problem["fit_intercept"])

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Trivial case: no samples
        if n_samples == 0:
            coef = [0.0] * n_features
            return {"coef": coef, "intercept": [0.0], "predictions": []}

        # Variable counts
        num_beta = n_features
        num_intercept = 1 if fit_intercept else 0
        num_u = n_samples
        num_v = n_samples
        Nvar = num_beta + num_intercept + num_u + num_v

        # Objective: minimize tau * u + (1-tau) * v
        c = np.zeros(Nvar, dtype=float)
        offset_u = num_beta + num_intercept
        c[offset_u: offset_u + num_u] = tau
        c[offset_u + num_u: offset_u + num_u + num_v] = 1.0 - tau

        # Build sparse equality constraint A_eq * z = b_eq
        # A_eq = [ X | 1(if intercept) | I_n | -I_n ]
        blocks = []
        if n_features > 0:
            blocks.append(sp.csr_matrix(X))
        else:
            blocks.append(sp.csr_matrix((n_samples, 0)))
        if fit_intercept:
            blocks.append(sp.csr_matrix(np.ones((n_samples, 1), dtype=float)))
        I = sp.identity(n_samples, format="csr", dtype=float)
        blocks.append(I)      # u variables
        blocks.append(-I)     # v variables

        A_eq = sp.hstack(blocks, format="csr")
        b_eq = y.astype(float)

        # Bounds: betas & intercept free, u/v >= 0
        bounds = [(None, None)] * (num_beta + num_intercept) + [(0.0, None)] * (num_u + num_v)

        # Solve LP using SciPy's HiGHS backend (fast)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            # Fallback to other solvers if needed
            for method in ("highs-ds", "highs-ipm", "interior-point", "revised simplex"):
                try:
                    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
                except Exception:
                    res = None
                if res is not None and getattr(res, "success", False):
                    break
            if res is None or not getattr(res, "success", False):
                raise RuntimeError("Quantile regression LP failed: " + (res.message if res is not None else "unknown"))

        z = res.x
        beta = z[:num_beta] if num_beta > 0 else np.zeros(0, dtype=float)
        idx = num_beta
        intercept_val = 0.0
        if fit_intercept:
            intercept_val = float(z[idx])
            idx += 1

        predictions = (X.dot(beta) + intercept_val).tolist()

        return {
            "coef": beta.tolist(),
            "intercept": [float(intercept_val)],
            "predictions": predictions,
        }