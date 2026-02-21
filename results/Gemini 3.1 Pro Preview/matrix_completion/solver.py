import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs):
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]

        mask = np.zeros((n, n), dtype=bool)
        mask[inds[:, 0], inds[:, 1]] = True

        B = cp.Variable((n, n), pos=True)
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        
        constraints = [
            cp.prod(B[~mask]) == 1.0,
            B[mask] == a,
        ]

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
            "optimal_value": result,
        }