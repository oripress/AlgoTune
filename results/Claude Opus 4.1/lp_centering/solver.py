import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def __init__(self):
        # Pre-configure solver settings for speed
        self.solver_options = {
            'verbose': False,
            'max_iter': 1000,
            'abstol': 1e-7,
            'reltol': 1e-7,
            'feastol': 1e-7
        }
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the LP centering problem using CVXPY with optimized settings.
        
        minimize    c^T x - sum(log(x_i))
        subject to  Ax = b
        """
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        n = len(c)
        
        # Create CVXPY problem
        x = cp.Variable(n)
        objective = cp.Minimize(c.T @ x - cp.sum(cp.log(x)))
        constraints = [A @ x == b]
        
        prob = cp.Problem(objective, constraints)
        
        # Try different solvers for speed
        try:
            # ECOS is usually fast for these problems
            prob.solve(solver=cp.ECOS, **self.solver_options)
            if prob.status == cp.OPTIMAL:
                return {"solution": x.value.tolist()}
        except:
            pass
        
        try:
            # Fall back to SCS if ECOS fails
            prob.solve(solver=cp.SCS, **self.solver_options)
            if prob.status == cp.OPTIMAL:
                return {"solution": x.value.tolist()}
        except:
            pass
        
        # Final fallback to CLARABEL (reference solver)
        prob.solve(solver=cp.CLARABEL)
        return {"solution": x.value.tolist()}