import math
from typing import Any
import cvxpy as cp
import numpy as np
from scipy.special import xlogy

class Solver:
    def __init__(self):
        # Precompute log(2) once
        self.ln2 = math.log(2)
    
    def solve(self, problem: dict) -> dict:
        P = np.array(problem["P"], dtype=np.float64)
        m, n = P.shape
        
        # Validate input - use faster check
        if not np.allclose(P.sum(axis=0), 1.0, atol=1e-9):
            return None
        
        # Precompute c vector
        c = np.sum(xlogy(P, P), axis=0) / self.ln2
        
        # Create optimization variable
        x = cp.Variable(n)
        
        # Define objective directly without intermediate variable
        # mutual_information = c @ x + cp.sum(cp.entr(P @ x) / ln2)
        y = P @ x
        objective = cp.Maximize(c @ x + cp.sum(cp.entr(y)) / self.ln2)
        
        # Define constraints
        constraints = [cp.sum(x) == 1, x >= 0]
        
        # Create and solve problem with specific solver
        prob = cp.Problem(objective, constraints)
        
        # Use ECOS solver which is typically faster for this type of problem
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, max_iters=100)
        except:
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or prob.value is None:
            return None
        
        return {"x": x.value.tolist(), "C": prob.value}