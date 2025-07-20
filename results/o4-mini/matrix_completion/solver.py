import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]

        # Build list of unknown indices
        xx, yy = np.meshgrid(np.arange(n), np.arange(n))
        allinds = np.vstack((yy.flatten(), xx.flatten())).T
        # Mask positions already observed
        mask_known = ((allinds[None, :, :] == inds[:, None, :]).all(2)).any(0)
        otherinds = allinds[~mask_known]

        # Define CVXPY variable and objective
        B = cp.Variable((n, n), pos=True)
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints: observed entries fixed, product of unknowns equals 1
        constraints = [B[inds[:, 0], inds[:, 1]] == a]
        if otherinds.shape[0] > 0:
            constraints.append(cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0)

        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except cp.SolverError:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or B.value is None:
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }