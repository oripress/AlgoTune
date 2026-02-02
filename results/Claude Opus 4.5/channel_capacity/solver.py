import numpy as np
from scipy.special import xlogy
import cvxpy as cp
import math

class Solver:
    def solve(self, problem, **kwargs):
        P = np.array(problem["P"], dtype=np.float64)
        m, n = P.shape
        
        ln2 = math.log(2)
        
        x = cp.Variable(n)
        y = P @ x
        c = np.sum(xlogy(P, P), axis=0) / ln2
        
        mutual_information = c @ x + cp.sum(cp.entr(y) / ln2)
        objective = cp.Maximize(mutual_information)
        constraints = [cp.sum(x) == 1, x >= 0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if prob.value is None:
            return None
        
        return {"x": x.value.tolist(), "C": prob.value}