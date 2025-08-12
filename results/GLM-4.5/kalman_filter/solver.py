import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """Solve Kalman filtering problem with optimized CVXPY formulation."""
        # Extract problem parameters
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        # Create optimization variables - only x and w, eliminate v
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        # Eliminate v using v = y - C @ x (only for x[0] to x[N-1])
        
        # Compute v from measurement equation: v = y - C @ x
        v = y - (C @ x[:N, :].T).T
        
        # Objective: minimize sum of squared noises
        objective = cp.Minimize(cp.sum_squares(w) + tau * cp.sum_squares(v))
        
        # Build constraints
        constraints = [x[0] == x0]
        
        # Dynamics constraints
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
        
        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        # Check if solution is valid
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        # Compute v_hat from the solution
        v_hat = y - (C @ x.value[:N, :].T).T
        
        # Return solution
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v_hat.tolist(),
        }