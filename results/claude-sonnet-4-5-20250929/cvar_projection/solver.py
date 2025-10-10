import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        self.beta = 0.95
        self.kappa = 1.0
        self.cached_prob = None
        self.cached_dims = None
        self.x_var = None
    
    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set.
        
        :param problem: Dictionary containing the point to project, loss scenarios, and parameters
        :return: Dictionary containing the projected point
        """
        # Extract problem data
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))
        
        n_scenarios, n_dims = A.shape
        k = int((1 - beta) * n_scenarios)
        alpha = kappa * k
        
        # Use CLARABEL with very relaxed tolerances for max speed
        try:
            x = cp.Variable(n_dims)
            objective = cp.Minimize(cp.sum_squares(x - x0))
            constraints = [cp.sum_largest(A @ x, k) <= alpha]
            prob = cp.Problem(objective, constraints)
            
            prob.solve(
                solver=cp.CLARABEL, 
                verbose=False, 
                tol_gap_abs=1e-4,
                tol_gap_rel=1e-4,
                tol_feas=1e-4,
                tol_infeas_abs=1e-4,
                tol_infeas_rel=1e-4,
                max_iter=50
            )
            
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                return {"x_proj": x.value.tolist()}
        except:
            pass
        
        return {"x_proj": []}