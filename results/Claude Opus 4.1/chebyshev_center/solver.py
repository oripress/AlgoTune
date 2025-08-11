import numpy as np
from scipy.optimize import linprog
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem using scipy's linprog.
        
        The problem is to maximize r subject to:
        a_i^T x_c + r ||a_i||_2 <= b_i for all i
        r >= 0
        
        We reformulate as minimization: minimize -r
        """
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        m, n = a.shape
        
        # Compute norms of each row of a
        a_norms = np.linalg.norm(a, axis=1)
        
        # Variables are [x_c (n dims), r (1 dim)]
        # Objective: minimize -r (maximize r)
        c = np.zeros(n + 1)
        c[-1] = -1  # coefficient for r
        
        # Inequality constraints: a_i^T x_c + r ||a_i||_2 <= b_i
        # Rewritten as: A_ub @ [x_c; r] <= b_ub
        A_ub = np.hstack([a, a_norms.reshape(-1, 1)])
        b_ub = b
        
        # Bounds: r >= 0, x_c unbounded
        bounds = [(None, None)] * n + [(0, None)]
        
        # Solve the LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            raise ValueError("Optimization failed")
        
        # Extract x_c (first n components)
        x_c = result.x[:n]
        
        return {"solution": x_c.tolist()}