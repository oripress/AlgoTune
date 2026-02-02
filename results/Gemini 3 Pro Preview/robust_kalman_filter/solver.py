from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using the Huber loss function.
        Optimized CVXPY implementation:
        - Vectorized constraints
        - Elimination of explicit 'v' variables to reduce problem size
        """
        # Extract problem data
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
        # We only keep x and w. v is implicit in the objective.
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")

        # Objective
        # Process noise: sum ||w_t||^2
        process_noise_term = cp.sum_squares(w)

        # Measurement noise: sum tau * phi(||y_t - C x_t||)
        # Vectorized: residuals = y - (x[:-1] @ C.T)
        # Note: x[:-1] corresponds to x_0 ... x_{N-1}
        residuals = y - x[:-1] @ C.T
        
        # Norm of each row (time step)
        res_norms = cp.norm(residuals, axis=1)
        
        # Huber loss on the norms
        measurement_noise_term = tau * cp.sum(cp.huber(res_norms, M))

        obj = cp.Minimize(process_noise_term + measurement_noise_term)

        # Constraints
        # Initial state
        constraints = [x[0] == x0]

        # Dynamics: x[1:] = x[:-1] @ A.T + w @ B.T
        constraints.append(x[1:] == x[:-1] @ A.T + w @ B.T)

        # Solve
        prob = cp.Problem(obj, constraints)
        
        # Try using CLARABEL with relaxed tolerance
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-4, tol_gap_rel=1e-4, verbose=False)
        except (cp.SolverError, ImportError, ValueError):
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except cp.SolverError:
                try:
                    prob.solve(verbose=False)
                except Exception:
                    return {"x_hat": [], "w_hat": [], "v_hat": []}
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Reconstruct v_hat
        # v_hat = y - C x_hat
        x_val = x.value
        v_val = y - x_val[:-1] @ C.T

        return {
            "x_hat": x_val.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v_val.tolist(),
        }