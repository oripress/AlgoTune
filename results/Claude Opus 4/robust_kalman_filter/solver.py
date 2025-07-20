from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using the Huber loss function.
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

        # Variables
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))
        v = cp.Variable((N, m))

        # Objective: minimize sum_{t=0}^{N-1}(||w_t||_2^2 + tau * phi(v_t))
        process_noise_term = cp.sum_squares(w)
        
        # Vectorized Huber norm calculation
        v_norms = cp.vstack([cp.norm(v[t, :]) for t in range(N)])
        measurement_noise_term = tau * cp.sum(cp.huber(v_norms, M))
        
        obj = cp.Minimize(process_noise_term + measurement_noise_term)

        # Constraints - vectorized formulation
        constraints = [x[0] == x0]  # Initial state
        
        # Stack all dynamics constraints at once
        constraints.append(x[1:] == x[:-1] @ A.T + w @ B.T)
        
        # Stack all measurement constraints at once
        constraints.append(y == x[:-1] @ C.T + v)

        # Solve the problem with CLARABEL solver (very fast for conic problems)
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except cp.SolverError:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }