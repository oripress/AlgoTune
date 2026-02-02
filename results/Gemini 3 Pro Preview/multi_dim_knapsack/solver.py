import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy

class Solver:
    def solve(self, problem, **kwargs):
        # Parse input
        if hasattr(problem, "value"):
            values = problem.value
            demands = problem.demand
            supply = problem.supply
        else:
            values, demands, supply = problem
        
        n = len(values)
        k = len(supply)
        
        # Objective: maximize sum(v_i * x_i) -> minimize sum(-v_i * x_i)
        c = -np.array(values, dtype=float)
        
        # Constraints: sum(d_ij * x_i) <= S_j
        # demand is n x k (rows are items)
        # We need A to be k x n (rows are constraints)
        # Note: demands might be a list of lists, so convert to numpy array
        A = np.array(demands).T
        b_u = np.array(supply, dtype=float)
        b_l = np.full_like(b_u, -np.inf)
        
        constraints = LinearConstraint(A, b_l, b_u)
        
        # Bounds for x: 0 <= x <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: 1 means integer
        integrality = np.ones(n)
        
        # Run MILP
        # options={'disp': False} to suppress output
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options={'disp': False})
        
        if res.success:
            # Extract indices of selected items
            # res.x contains the solution values
            selected = [i for i, x_val in enumerate(res.x) if x_val > 0.5]
            return selected
        else:
            return []