from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        """
        Solve the Chebyshev center problem using scipy's linprog.
        
        The problem is to maximize r subject to:
        a_i^T x + r ||a_i||_2 <= b_i for all i
        r >= 0
        
        We convert this to standard form for linprog (minimization):
        minimize -r
        """
        a = np.array(problem["a"])
        b = np.array(problem["b"])
        m, n = a.shape
        
        # Compute norms of each row of a
        a_norms = np.linalg.norm(a, axis=1)
        
        # Variables are [x1, x2, ..., xn, r]
        # Objective: minimize -r (to maximize r)
        c = np.zeros(n + 1)
        c[-1] = -1  # coefficient for r
        
        # Constraints: a_i^T x + r ||a_i||_2 <= b_i
        # In standard form: A_ub @ [x; r] <= b_ub
        A_ub = np.hstack([a, a_norms.reshape(-1, 1)])
        b_ub = b
        
        # Bounds: r >= 0, x unconstrained
        bounds = [(None, None)] * n + [(0, None)]
        
        # Solve the LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            raise ValueError("Optimization failed")
        
        # Extract the solution (first n components are x)
        x_solution = result.x[:n]
        
        return {"solution": x_solution.tolist()}