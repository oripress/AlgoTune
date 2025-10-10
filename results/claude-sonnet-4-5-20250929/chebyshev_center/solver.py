from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem as a standard LP using scipy's HiGHS solver.
        
        Reformulate as:
        minimize c^T @ [x_c; r] = -r
        subject to [a_i, ||a_i||_2] @ [x_c; r] <= b_i for all i
        and r >= 0
        """
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        n = a.shape[1]
        m = a.shape[0]
        
        # Pre-compute norms
        a_norms = np.linalg.norm(a, axis=1).reshape(-1, 1)
        
        # Construct the LP in standard form
        # Variables: [x_c (n dims), r (1 dim)]
        # Objective: minimize -r (to maximize r)
        c = np.zeros(n + 1)
        c[-1] = -1.0  # Coefficient for r in objective
        
        # Inequality constraints: [a_i, ||a_i||_2] @ [x_c; r] <= b_i
        A_ub = np.empty((m, n + 1), dtype=np.float64)
        A_ub[:, :n] = a
        A_ub[:, n] = a_norms.ravel()
        
        # Bounds: no bounds on x_c, r >= 0
        bounds = [(None, None)] * n + [(0, None)]
        
        # Solve using HiGHS
        result = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs')
        
        # Extract x_c (first n components)
        return {"solution": result.x[:n].tolist()}