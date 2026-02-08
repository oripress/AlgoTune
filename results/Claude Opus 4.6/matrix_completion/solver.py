import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]
        
        obs_rows = inds[:, 0]
        obs_cols = inds[:, 1]
        
        obs_mask = np.zeros((n, n), dtype=bool)
        obs_mask[obs_rows, obs_cols] = True
        free_ri, free_ci = np.where(~obs_mask)
        n_free = len(free_ri)
        
        B = cp.Variable((n, n), pos=True)
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        
        constraints = [B[obs_rows[i], obs_cols[i]] == a[i] for i in range(len(a))]
        
        if n_free > 0:
            prod_expr = B[free_ri[0], free_ci[0]]
            for k in range(1, n_free):
                prod_expr = prod_expr * B[free_ri[k], free_ci[k]]
            constraints.append(prod_expr == 1.0)
        
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except Exception:
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if B.value is None:
            return None
        
        return {
            "B": B.value.tolist(),
            "optimal_value": float(result),
        }