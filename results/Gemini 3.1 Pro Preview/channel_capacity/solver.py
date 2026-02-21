import numpy as np
import math
import cvxpy as cp
from scipy.special import xlogy

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        P = np.array(problem["P"])
        m, n = P.shape
        
        if n == 0 or m == 0:
            return None
            
        x = cp.Variable(n)
        y = P @ x
        
        # Compute c in nats to avoid division in the CVXPY expression tree
        c = np.sum(xlogy(P, P), axis=0)
        
        # Maximize mutual information in nats
        objective = cp.Maximize(c @ x + cp.sum(cp.entr(y)))
        constraints = [cp.sum(x) == 1, x >= 0]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            # Explicitly use ECOS to avoid solver selection overhead
            prob.solve(solver=cp.ECOS)
        except Exception:
            return None
            
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
            
        if prob.value is None:
            return None
            
        x_val = x.value
        x_val = np.maximum(x_val, 0)
        x_val /= np.sum(x_val)
        
        # Convert capacity from nats to bits
        C_bits = prob.value / math.log(2)
        
        return {"x": x_val.tolist(), "C": float(C_bits)}