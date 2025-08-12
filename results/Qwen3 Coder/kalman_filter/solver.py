import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """Solve the Kalman filtering problem using convex optimization."""
        try:
            A = np.array(problem["A"])
            B = np.array(problem["B"])
            C = np.array(problem["C"])
            y = np.array(problem["y"])
            x0 = np.array(problem["x_initial"])
            tau = float(problem["tau"])
            
            N, m = y.shape
            n = A.shape[1]  # Fixed dimension
            p = B.shape[1]
            
            # Create optimization variables
            x = cp.Variable((N + 1, n))
            w = cp.Variable((N, p))
            v = cp.Variable((N, m))
            
            # Formulate objective: minimize sum of squares
            obj = cp.Minimize(cp.sum_squares(w) + tau * cp.sum_squares(v))
            
            # Formulate constraints more efficiently
            constraints = [x[0] == x0]  # Initial state constraint
            
            # Vectorized constraints for better performance
            # Dynamics: x_{t+1} = A*x_t + B*w_t for all t
            # Measurements: y_t = C*x_t + v_t for all t
            for t in range(N):
                constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
                constraints.append(y[t] == C @ x[t] + v[t])
            
            # Solve the problem with specific solver for better performance
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.OSQP, verbose=False, max_iter=10000)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"x_hat": [], "w_hat": [], "v_hat": []}
            
            return {
                "x_hat": x.value.tolist(),
                "w_hat": w.value.tolist(),
                "v_hat": v.value.tolist(),
            }
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}