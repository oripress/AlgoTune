from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using a convex formulation
        with a Huber loss on measurement residuals.

        Parameters
        ----------
        problem : dict
            Dictionary containing system matrices, measurements, and parameters.

        Returns
        -------
        dict
            Dictionary with estimated state sequence ``x_hat``,
            process noise sequence ``w_hat`` and measurement noise
            sequence ``v_hat``.
        """
        # Extract data
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape                     # number of measurements, measurement dim
        n = A.shape[1]                     # state dimension
        p = B.shape[1]                     # process‑noise dimension

        # Decision variables: state and process noise
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))

        # Objective: quadratic process‑noise term + Huber measurement term using residuals
        proc_term = cp.sum_squares(w)
        # residuals y - C x[t]
        residuals = y - cp.matmul(x[:N], C.T)   # shape (N, m)
        meas_term = tau * cp.sum(cp.huber(cp.norm(residuals, axis=1), M))

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])   # dynamics
        # Solve the problem using a solver that supports Huber (ECOS)
        prob = cp.Problem(cp.Minimize(proc_term + meas_term), constraints)
        prob.solve(solver=cp.ECOS)

        # Check solution status
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Retrieve optimal values
        x_opt = x.value
        w_opt = w.value

        # Compute measurement noise v_hat from residuals y - C x
        v_opt = []
        for t in range(N):
            v_opt.append((y[t] - C @ x_opt[t]).tolist())

        return {
            "x_hat": x_opt.tolist(),
            "w_hat": w_opt.tolist(),
            "v_hat": v_opt,
        }