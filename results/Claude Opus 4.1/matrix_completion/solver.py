import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem):
        """
        Solves the Perron-Frobenius matrix completion problem.
        
        Args:
            problem: Dict containing inds, a, and n.
        
        Returns:
            Dict with B matrix and optimal_value.
        """
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]
        
        # Find missing indices
        xx, yy = np.meshgrid(np.arange(n), np.arange(n))
        allinds = np.vstack((yy.flatten(), xx.flatten())).T
        otherinds = allinds[~(allinds == inds[:, None]).all(2).any(0)]
        
        # Create CVXPY variable
        B = cp.Variable((n, n), pos=True)
        
        # Objective: minimize Perron-Frobenius eigenvalue
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        
        # Constraints
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a,
        ]
        
        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except:
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        
        if B.value is None:
            return None
        
        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }