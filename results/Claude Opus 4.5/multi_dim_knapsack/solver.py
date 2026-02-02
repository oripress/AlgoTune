from typing import Any
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        # Parse the problem
        if hasattr(problem, 'value'):
            value = problem.value
            demand = problem.demand
            supply = problem.supply
        else:
            value, demand, supply = problem
        
        n = len(value)
        if n == 0:
            return []
        
        k = len(supply)
        
        # Convert to numpy arrays for efficiency
        value_arr = np.array(value, dtype=np.float64)
        demand_arr = np.array(demand, dtype=np.float64)
        supply_arr = np.array(supply, dtype=np.float64)
        
        # Objective: maximize sum of values (scipy minimizes, so negate)
        c = -value_arr
        
        # Constraints: demand.T @ x <= supply
        # demand[i][j] is demand of item i for resource j
        # So for resource r: sum_i(x[i] * demand[i][r]) <= supply[r]
        # This is demand.T @ x <= supply where demand.T has shape (k, n)
        A = demand_arr.T  # Shape (k, n)
        
        # Bounds: 0 <= x <= 1, integers (binary)
        bounds = Bounds(0, 1)
        integrality = np.ones(n)  # All variables are integers (binary)
        
        constraints = LinearConstraint(A, -np.inf, supply_arr)
        
        result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if result.success:
            return [i for i in range(n) if result.x[i] > 0.5]
        return []