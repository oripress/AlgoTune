from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using the Huber loss function.
        
        :param problem: Dictionary with system matrices, measurements, and parameters
        :return: Dictionary with estimated states and noise sequences
        """
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
        
        # Variables: x[0]...x[N], w[0]...w[N-1], v[0]...v[N-1]
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")
        
        # Objective: minimize sum_{t=0}^{N-1}(||w_t||_2^2 + tau * phi(v_t))
        # Process noise (quadratic) - vectorized for efficiency
        process_noise_term = cp.sum(cp.square(w))  # More direct than sum_squares
        
        # Measurement noise (Huber) - vectorized for better performance
        # Use huber directly on the norm instead of norm of huber for better performance
        v_norms = cp.norm(v, axis=1)
        huber_expr = cp.huber(v_norms, M)
        measurement_noise_term = tau * cp.sum(huber_expr)
        
        obj = cp.Minimize(process_noise_term + measurement_noise_term)
        
        # Constraints - optimized vectorized form
        constraints = []
        
        # Initial state constraint - ensure it's properly enforced
        constraints.append(x[0, :] == x0)
        
        # Vectorized dynamics constraints: x[t+1] = A*x[t] + B*w[t]
        # Stack all dynamics constraints at once for better performance
        dynamics_lhs = x[1:N+1, :]  # x[1], ..., x[N]
        dynamics_rhs = x[:N, :] @ A.T + w @ B.T  # A*x[t] + B*w[t] for all t
        constraints.append(dynamics_lhs == dynamics_rhs)
        
        # Vectorized measurement constraints: y[t] = C*x[t] + v[t]  
        measurement_lhs = y  # y[0], ..., y[N-1]
        measurement_rhs = x[:N, :] @ C.T + v  # C*x[t] + v[t] for all t
        constraints.append(measurement_lhs == measurement_rhs)
        
        # Solve the problem with optimized settings
        prob = cp.Problem(obj, constraints)
        try:
            # Use OSQP solver with optimized settings for speed (often faster than ECOS for this problem)
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=100)
        except (cp.SolverError, Exception):
            # Fallback to ECOS if OSQP fails
            try:
                prob.solve(solver=cp.ECOS, verbose=False, max_iters=50, abstol=1e-6, reltol=1e-6)
            except Exception:
                # Final fallback to default solver
                try:
                    prob.solve(verbose=False, max_iters=50)
                except Exception:
                    return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }