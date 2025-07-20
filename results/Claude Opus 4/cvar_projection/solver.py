import numpy as np
import cvxpy as cp

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set using an optimized approach.
        """
        # Extract problem data
        x0 = np.array(problem["x0"], dtype=np.float64)
        A = np.array(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 1.0))
        
        n_scenarios, n_dims = A.shape
        k = int((1 - beta) * n_scenarios)
        
        # First check if x0 is already feasible
        losses_x0 = A @ x0
        if k == 0:
            cvar_x0 = np.max(losses_x0)
        else:
            sorted_losses = np.sort(losses_x0)[-k:]
            cvar_x0 = np.mean(sorted_losses)
        
        if cvar_x0 <= kappa + 1e-8:
            # x0 is already feasible
            return {"x_proj": x0.tolist()}
        
        # Use ECOS solver which is faster than default
        x = cp.Variable(n_dims)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        
        # CVaR constraint using sum_largest
        alpha = kappa * k
        constraints = [cp.sum_largest(A @ x, k) <= alpha]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use ECOS solver with optimized settings
            prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-7, reltol=1e-6, feastol=1e-7)
            
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                return {"x_proj": x.value.tolist()}
            else:
                # If solver fails, return empty list
                return {"x_proj": []}
        except Exception as e:
            # Return empty list on any error
            return {"x_proj": []}