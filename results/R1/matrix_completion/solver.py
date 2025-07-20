import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """Solves the Perron-Frobenius matrix completion using CVXPY."""
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]
        
        xx, yy = np.meshgrid(np.arange(n), np.arange(n))
        allinds = np.vstack((yy.flatten(), xx.flatten())).T
        otherinds = allinds[~(allinds == inds[:, None]).all(2).any(0)]
        
        # Define CVXPY Variables
        B = cp.Variable((n, n), pos=True)
        
        # Define Objective
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        
        # Define Constraints
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a,
        ]
        
        # Solve Problem
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except (cp.SolverError, Exception):
            return None
        
        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if B.value is None:
            return None
        
        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }