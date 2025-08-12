from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using the Huber loss function.
        Optimized implementation with vectorized Huber calculation and fine-tuned solver parameters.

        :param problem: Dictionary with system matrices, measurements, and parameters
        :return: Dictionary with estimated states and noise sequences
        """
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        C = np.array(problem["C"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        x0 = np.array(problem["x_initial"], dtype=np.float64)
        tau = float(problem["tau"])
        M = float(problem["M"])
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        # Variables: x[0]...x[N], w[0]...w[N-1], v[0]...v[N-1]
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")
        
        # Objective: minimize sum_{t=0}^{N-1}(||w_t||_2^2 + tau * phi(v_t))
        # Process noise (quadratic)
        process_noise_term = cp.sum_squares(w)
        
        # Measurement noise (Huber) - vectorized for efficiency
        v_norms = cp.norm(v, axis=1)
        measurement_noise_term = tau * cp.sum(cp.huber(v_norms, M))
        
        obj = cp.Minimize(process_noise_term + measurement_noise_term)
        
        # Constraints
        constraints = [x[0] == x0]  # Initial state
        
        # Add dynamics and measurement constraints
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])  # Dynamics
            constraints.append(y[t] == C @ x[t] + v[t])  # Measurement
        
        # Solve the problem with optimized settings
        prob = cp.Problem(obj, constraints)
        try:
            # Use SCS with fine-tuned parameters for fastest convergence
            prob.solve(verbose=False, solver=cp.SCS, max_iters=500, eps=1e-5, scale=2.5, acceleration_lookback=6, normalize=True, use_indirect=False)
        except cp.SolverError:
            try:
                # Fallback to ECOS with optimized settings
                prob.solve(verbose=False, solver=cp.ECOS, max_iters=150, feastol=1e-5, reltol=1e-5, abstol=1e-5)
            except cp.SolverError:
                # Last resort: default solver
                prob.solve(verbose=False)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }