import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        self.prev_solution = None
        
    def solve(self, problem, **kwargs) -> dict:
        # Extract parameters
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = problem["beta"]
        kappa = problem["kappa"]
        n_scenarios, n_dims = A.shape

        # Precompute constants - ensure k is at least 1
        k = max(1, min(n_scenarios, int((1 - beta) * n_scenarios)))
        alpha = kappa * k

        # Precompute only essential values
        x0_squared = np.dot(x0, x0)

        # Define variables with warm-start
        x = cp.Variable(n_dims)
        if self.prev_solution is not None:
            x.value = self.prev_solution

        # CVaR constraint using sum_largest
        losses = A @ x
        constraint = [cp.sum_largest(losses, k) <= alpha]

        # Efficient objective: (1/2)||x - x0||² = (1/2)xᵀx - x0ᵀx + (1/2)x0ᵀx0
        objective = cp.Minimize(0.5 * cp.sum_squares(x) - x0 @ x + 0.5 * x0_squared)

        # Solve with highly optimized ECOS parameters
        prob = cp.Problem(objective, constraint)
        try:
            # Aggressive settings for maximum speed
            prob.solve(solver=cp.ECOS, 
                       abstol=1e-4, 
                       reltol=1e-4, 
                       max_iters=500,
                       feastol=1e-4,
                       reltol_inacc=1e-3)

            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                self.prev_solution = x.value
                return {"x_proj": x.value.tolist()}
            
            # If ECOS fails, try OSQP as fallback
            prob.solve(solver=cp.OSQP, verbose=False, 
                       eps_abs=1e-4, eps_rel=1e-4, 
                       max_iter=10000, warm_start=True)
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                self.prev_solution = x.value
                return {"x_proj": x.value.tolist()}
            
            return {"x_proj": []}
        except Exception as e:
            return {"x_proj": []}