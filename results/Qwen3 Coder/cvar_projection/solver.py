import numpy as np
from scipy.optimize import minimize

class Solver:
    def __init__(self):
        # Default parameters
        self.beta = 0.95
        self.kappa = 1.0
    
    def solve(self, problem, **kwargs) -> dict:
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
        
        # Define objective function: ||x - x0||^2
        def objective(x):
            return np.sum((x - x0)**2)
            
        # Define gradient of objective
        def objective_grad(x):
            return 2 * (x - x0)
        
        # Define CVaR constraint function
        # CVaR_beta(Ax) = (1/k) * sum of k largest losses <= kappa
        # where k = (1-beta) * n_scenarios
        k = int((1 - beta) * n_scenarios)
        
        def cvar_constraint(x):
            losses = A @ x
            
            # Handle edge case when k=0
            if k == 0:
                # When k=0, no constraint is needed, so return a value that satisfies constraint
                return 1.0
                
            # Get the k largest losses (worst case scenarios)
            sorted_losses = np.sort(losses)[-k:]
            
            # Compute CVaR: average of k largest losses
            cvar_value = np.sum(sorted_losses) / k
            
            
            # Constraint: CVaR <= kappa, so we return kappa - cvar_value
            return kappa - cvar_value

        # Define constraint dictionary for scipy.optimize
        constraint = {'type': 'ineq', 'fun': cvar_constraint}
        
        # Initial guess
        x0_guess = x0.copy()
        
        # Solve optimization problem
        result = minimize(
            objective,
            x0_guess,
            method='SLSQP',
            jac=objective_grad,
            constraints=constraint,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        # Return solution
        return {"x_proj": result.x.tolist()}