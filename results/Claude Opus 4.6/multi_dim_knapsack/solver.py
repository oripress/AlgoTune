import numpy as np
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Solve Multi-Dimensional Knapsack using HiGHS MIP solver."""
        if isinstance(problem, (list, tuple)):
            value, demand, supply = problem
        else:
            value = problem.value
            demand = problem.demand
            supply = problem.supply
        
        n = len(value)
        k = len(supply)
        
        if n == 0:
            return []
        
        # Convert to numpy arrays
        value_arr = np.array(value, dtype=np.float64)
        demand_arr = np.array(demand, dtype=np.float64)
        supply_arr = np.array(supply, dtype=np.float64)
        
        # Use scipy's milp with HiGHS backend
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Maximize value = minimize negative value
        c = -value_arr
        
        # Constraints: demand^T x <= supply
        # demand_arr is n x k, we need A x <= supply
        # A[r][i] = demand[i][r], so A = demand_arr.T
        A = demand_arr.T  # k x n
        
        constraints = LinearConstraint(A, ub=supply_arr)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)  # All variables are integers (binary)
        
        result = milp(
            c=c, 
            constraints=constraints, 
            bounds=bounds, 
            integrality=integrality,
        )
        
        if result.success:
            x = result.x
            selected = [i for i in range(n) if x[i] > 0.5]
            return selected
        
        return []