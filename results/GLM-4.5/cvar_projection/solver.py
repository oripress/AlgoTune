import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set.
        
        :param problem: Dictionary containing the point to project, loss scenarios, and parameters
        :return: Dictionary containing the projected point
        """
        # Extract problem data
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])
        
        n_scenarios, n_dims = A.shape
        
        # Define variables
        x = cp.Variable(n_dims)
        
        # Define objective: minimize distance to x0
        objective = cp.Minimize(cp.sum_squares(x - x0))
        
        # Add CVaR constraint using dual formulation for better performance
        k = int((1 - beta) * n_scenarios)
        
        # Pre-compute 1/k for efficiency
        inv_k = 1.0 / k
        
        # Dual formulation of CVaR constraint
        losses = A @ x
        t = cp.Variable()
        u = cp.Variable(n_scenarios)
        
        # Dual CVaR constraint: t + (1/k) * sum(u_i) <= kappa, with u_i >= losses_i - t, u_i >= 0
        constraints = [
            t + inv_k * cp.sum(u) <= kappa,
            u >= losses - t,
            u >= 0
        ]
        
        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(
                solver=cp.ECOS, 
                verbose=False, 
                max_iters=25,
                feastol=2.5e-2,
                reltol=2.5e-2,
                abstol=2.5e-2
            )
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                return {"x_proj": []}
            
            return {"x_proj": x.value.tolist()}
            
        except cp.SolverError as e:
            return {"x_proj": []}
        except Exception as e:
            return {"x_proj": []}