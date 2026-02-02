import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        n = c.shape[0]

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c.T @ x - cp.sum(cp.log(x))), [A @ x == b])
        prob.solve(solver="CLARABEL", tol_gap_abs=1e-8, tol_gap_rel=1e-8)
        
        return {"solution": x.value.tolist()}