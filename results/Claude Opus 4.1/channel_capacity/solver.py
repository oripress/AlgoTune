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
        
        # Validate input
        if not np.allclose(np.sum(P, axis=0), 1, atol=1e-6):
            return None
        
        # Create optimization variable
        x = cp.Variable(n)
        
        # Calculate output distribution
        y = P @ x
        
        # Precompute c vector (negative entropy terms)
        c = np.sum(xlogy(P, P), axis=0) / self.ln2
        
        # Mutual information objective
        mutual_information = c @ x + cp.sum(cp.entr(y) / self.ln2)
        
        # Set up optimization problem
        objective = cp.Maximize(mutual_information)
        constraints = [cp.sum(x) == 1, x >= 0]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use ECOS solver with relaxed tolerances for speed
            prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, max_iters=50)
        except (cp.SolverError, Exception):
            return None
        
        # Check if solution is valid
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if prob.value is None:
            return None
        
        return {"x": x.value.tolist(), "C": prob.value}