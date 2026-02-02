import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = int(problem["n"])
        
        # Create observed mask - more efficient than reference
        observed = np.zeros((n, n), dtype=bool)
        if len(inds) > 0:
            observed[inds[:, 0], inds[:, 1]] = True
        
        # Find unobserved indices
        unobs_rows, unobs_cols = np.where(~observed)
        
        # Define optimization variable
        B = cp.Variable((n, n), pos=True)
        
        # Define constraints
        constraints = []
        
        # Observed entries constraint
        if len(inds) > 0:
            constraints.append(B[inds[:, 0], inds[:, 1]] == a)
        
        # Product constraint for unobserved entries
        if len(unobs_rows) > 0:
            constraints.append(cp.prod(B[unobs_rows, unobs_cols]) == 1.0)
        
        # Define objective
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        
        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except Exception:
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or B.value is None:
            return None
        
        return {
            "B": B.value.tolist(),
            "optimal_value": float(result),
        }