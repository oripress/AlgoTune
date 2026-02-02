import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Print installed solvers for debugging
        # print(f"Installed solvers: {cp.installed_solvers()}")
        
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 0.0))

        n_scenarios, n_dims = A.shape
        k = int((1 - beta) * n_scenarios)
        
        # If k=0, the constraint is 0 <= 0, so x = x0
        if k == 0:
            return {"x_proj": x0.tolist()}

        # Check if x0 already satisfies the constraint
        # Calculate CVaR of x0
        losses = A @ x0
        # sum of k largest
        if k > 0:
            top_k_losses = np.sort(losses)[-k:]
            cvar_val = np.sum(top_k_losses) / k
        else:
            cvar_val = 0.0
            
        if cvar_val <= kappa + 1e-6:
             return {"x_proj": x0.tolist()}

        # Define variables
        x = cp.Variable(n_dims)
        
        # Objective
        objective = cp.Minimize(cp.sum_squares(x - x0))
        
        # Constraint
        # sum_largest(Ax, k) <= k * kappa
        constraints = [cp.sum_largest(A @ x, k) <= k * kappa]
        
        prob = cp.Problem(objective, constraints)
        
        # Try solving with OSQP if available, else default
        try:
            if 'OSQP' in cp.installed_solvers():
                prob.solve(solver=cp.OSQP, verbose=False)
            else:
                prob.solve(verbose=False)
        except Exception:
            return {"x_proj": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_proj": []}

        return {"x_proj": x.value.tolist()}