import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

class Solver:
    def solve(self, problem, **kwargs):
        if isinstance(problem, (list, tuple)):
            value, demand, supply = problem
        else:
            value = problem.value
            demand = problem.demand
            supply = problem.supply
            
        c = -np.array(value, dtype=np.float64)
        A = np.array(demand, dtype=np.float64).T
        b = np.array(supply, dtype=np.float64)
        
        constraints = LinearConstraint(A, -np.inf, b)
        integrality = np.ones_like(c)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = np.round(res.x).astype(int)
            return [i for i, val in enumerate(x) if val == 1]
        return []