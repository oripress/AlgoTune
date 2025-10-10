import math
import cvxpy as cp
import numpy as np
from scipy.special import xlogy

class Solver:
    def solve(self, problem: dict) -> dict:
        P = np.array(problem["P"])
        m, n = P.shape
        
        if m <= 0 or n <= 0:
            return None
        if not np.allclose(np.sum(P, axis=0), 1):
            return None

        x = cp.Variable(shape=n, name="x")
        y = P @ x
        c = np.sum(xlogy(P, P), axis=0) / math.log(2)
        mutual_information = c @ x + cp.sum(cp.entr(y) / math.log(2))
        objective = cp.Maximize(mutual_information)
        constraints = [cp.sum(x) == 1, x >= 0]

        prob = cp.Problem(objective, constraints)
        try:
            # Use ECOS with more aggressive settings for speed
            prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-7, reltol=1e-7, 
                      feastol=1e-7, max_iters=50)
        except:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if prob.value is None:
            return None

        return {"x": x.value.tolist(), "C": prob.value}