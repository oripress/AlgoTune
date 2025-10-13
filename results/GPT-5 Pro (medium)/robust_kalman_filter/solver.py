from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using a vectorized convex formulation.
        Eliminates measurement noise variables v via substitution to reduce problem size.

        :param problem: Dictionary with system matrices, measurements, and parameters
        :return: Dictionary with estimated states and noise sequences
        """
        try:
            A = np.asarray(problem["A"], dtype=float)
            B = np.asarray(problem["B"], dtype=float)
            C = np.asarray(problem["C"], dtype=float)
            y = np.asarray(problem["y"], dtype=float)
            x0 = np.asarray(problem["x_initial"], dtype=float)
            tau = float(problem["tau"])
            M = float(problem["M"])
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Dimensions
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Quick sanity on shapes (minimal checks to avoid solver crashes)
        if A.shape != (n, n) or B.shape[0] != n or C.shape[1] != n or x0.shape != (n,):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Decision variables: x[0..N], w[0..N-1]; eliminate v via substitution
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))

        # Residuals for measurement: v_t = y_t - C x_t
        # Vectorized dynamics and measurement residuals
        # Dynamics: x[1:] == x[:-1] @ A.T + w @ B.T
        constraints = [x[0] == x0]
        if N > 0:
            constraints.append(x[1:] == x[:-1] @ A.T + w @ B.T)

        # Objective:
        #   sum_squares(w) + tau * sum_t huber( ||y_t - C x_t||_2, M )
        if N > 0:
            residual = y - x[:-1] @ C.T  # shape (N, m)
            meas_term = tau * cp.sum(cp.huber(cp.norm(residual, axis=1), M))
        else:
            meas_term = 0.0

        obj = cp.Minimize(cp.sum_squares(w) + meas_term)

        prob = cp.Problem(obj, constraints)
        try:
            # ECOS handles SOC constraints efficiently; warm_start may help on repeated calls.
            prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        x_val = x.value
        w_val = w.value
        if x_val is None or w_val is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Recover v from residual definition
        if N > 0:
            v_val = y - x_val[:-1] @ C.T
        else:
            v_val = np.empty((0, m))

        # Ensure finiteness
        if not (np.isfinite(x_val).all() and np.isfinite(w_val).all() and np.isfinite(v_val).all()):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x_val.tolist(),
            "w_hat": w_val.tolist(),
            "v_hat": v_val.tolist(),
        }