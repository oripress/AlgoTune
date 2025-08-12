import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """Solve the robust Kalman filtering problem using the Huber loss function."""
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective
        process_noise_term = cp.sum_squares(w)
        
        # Vectorized Huber norm calculation
        residuals = cp.norm(v, axis=1)
        measurement_noise_term = tau * cp.sum(cp.huber(residuals, M))
        
        obj = cp.Minimize(process_noise_term + measurement_noise_term)

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        # Solve with SCS solver
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.SCS, eps=1e-5, max_iters=10000)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }