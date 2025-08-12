import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        inds = np.array(problem["inds"], dtype=int)
        a = np.array(problem["a"], dtype=float)
        n = int(problem["n"])

        # Create grid of all indices
        xx, yy = np.meshgrid(np.arange(n), np.arange(n))
        allinds = np.vstack((yy.flatten(), xx.flatten())).T

        # Identify missing entries (otherinds)
        mask = (allinds[:, None] == inds[None, :, :]).all(axis=2).any(axis=1)
        otherinds = allinds[~mask]

        # Define variable
        B = cp.Variable((n, n), pos=True)

        # Objective: minimize Perron-Frobenius eigenvalue
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints: fixed entries and product of missing = 1
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a,
        ]

        # Solve GP
        prob = cp.Problem(objective, constraints)
        result = prob.solve(gp=True)

        # Return solution
        return {"B": B.value.tolist(), "optimal_value": result}