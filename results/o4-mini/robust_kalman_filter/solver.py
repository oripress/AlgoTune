from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Unpack and convert
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        C = np.array(problem["C"], dtype=float)
        y = np.array(problem["y"], dtype=float)      # shape (N, m)
        x0 = np.array(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])

        # Dimensions
        N, m = y.shape
        n = A.shape[0]
        p = B.shape[1]

        # Transposed state and process‐noise variables
        # X has shape (n, N+1), W has shape (p, N)
        X = cp.Variable((n, N + 1))
        W = cp.Variable((p, N))

        # Constraints: initial state and dynamics for all t in one vectorized form
        constraints = [
            X[:, 0] == x0,
            X[:, 1:] == A @ X[:, :-1] + B @ W
        ]

        # Residuals for measurement noise: shape (m, N)
        residuals = C @ X[:, :-1] - y.T

        # Objective terms
        process_term = cp.sum_squares(W)  # sum ||w_t||^2
        # Huber penalty on each time‐step measurement residual norm
        res_norms = cp.norm(residuals, axis=0)  # length‐N vector
        meas_term = tau * cp.sum(cp.huber(res_norms, M))

        obj = cp.Minimize(process_term + meas_term)

        # Solve with ECOS
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True)
        except (cp.SolverError, Exception):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Extract and return in the expected format
        Xv = X.value      # shape n x (N+1)
        Wv = W.value      # shape p x N
        x_hat = Xv.T.tolist()
        w_hat = Wv.T.tolist()
        v_hat = (y - (C @ Xv[:, :-1]).T).tolist()

        return {
            "x_hat": x_hat,
            "w_hat": w_hat,
            "v_hat": v_hat,
        }