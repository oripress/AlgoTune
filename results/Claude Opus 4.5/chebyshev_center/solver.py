from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Chebyshev center problem using linear programming.
        """
        a = np.asarray(problem["a"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        m, n = a.shape
        
        # Compute norms of each row of a
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        
        # Variables: [x_1, ..., x_n, r]
        # Objective: maximize r => minimize -r
        c = np.zeros(n + 1)
        c[-1] = -1.0
        
        # Constraints: A @ x + r * norms <= b
        A_ub = np.hstack([a, norms])
        
        # Use numpy arrays for bounds for efficiency
        lb = np.full(n + 1, -np.inf)
        ub = np.full(n + 1, np.inf)
        lb[n] = 0  # r >= 0
        
        result = linprog(c, A_ub=A_ub, b_ub=b, bounds=np.column_stack([lb, ub]), method='highs')
        
        return {"solution": result.x[:n].tolist()}