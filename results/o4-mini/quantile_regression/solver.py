import numpy as np
from scipy import sparse
from scipy.optimize import linprog

class Solver:
    def solve(self, problem, **kwargs):
        # Parse input
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        tau = float(problem["quantile"])
        fit_intercept = bool(problem.get("fit_intercept", True))
        n, p = X.shape

        # Build the equality constraint A_eq x = b_eq in sparse CSC
        # Variable order: [beta (p), intercept? (1), u (n), v (n)]
        blocks = []
        # -X * beta
        blocks.append(sparse.csc_matrix(-X))
        # -1 * intercept
        if fit_intercept:
            blocks.append(sparse.csc_matrix(-np.ones((n, 1), dtype=float)))
        # u >= 0 (positive residuals)
        I = sparse.eye(n, format="csc")
        blocks.append(I)
        # v >= 0 (negative residuals)
        blocks.append(-I)
        A_eq = sparse.hstack(blocks, format="csc")
        b_eq = -y

        # Objective: tau * sum(u) + (1-tau) * sum(v)
        num_vars = p + (1 if fit_intercept else 0) + 2 * n
        c = np.zeros(num_vars, dtype=float)
        offset = p + (1 if fit_intercept else 0)
        c[offset : offset + n] = tau
        c[offset + n : offset + 2 * n] = 1.0 - tau

        # Bounds: beta and intercept free, u & v >= 0
        bounds = [(None, None)] * (p + (1 if fit_intercept else 0)) + [(0.0, None)] * (2 * n)

        # Solve LP via HiGHS dual simplex, skip presolve for lower overhead
        res = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs-ds",
            options={"presolve": False},
        )
        x = res.x

        # Extract solution
        coef = x[:p].tolist()
        intercept_val = x[p] if fit_intercept else 0.0
        intercept = [float(intercept_val)]
        preds = (X.dot(x[:p]) + intercept_val).tolist()

        return {"coef": coef, "intercept": intercept, "predictions": preds}