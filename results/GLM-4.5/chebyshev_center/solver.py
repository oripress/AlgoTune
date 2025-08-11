from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem using optimized scipy.optimize.linprog.
        
        :param problem: A dictionary of the Chebyshev center problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the Chebyshev center problem.
        """
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        m, n = a.shape
        
        # Pre-compute norms using more efficient method
        a_norms = np.sqrt(np.sum(a**2, axis=1))
        
        # Formulate as LP: maximize r
        # Variables: [x1, x2, ..., xn, r]
        # Objective: c = [0, 0, ..., 0, 1] (maximize r)
        c = np.zeros(n + 1, dtype=np.float64)
        c[-1] = -1  # linprog minimizes, so we minimize -r to maximize r
        
        # Build constraint matrix more efficiently
        A_ub = np.empty((m, n + 1), dtype=np.float64)
        A_ub[:, :n] = a
        A_ub[:, n] = a_norms
        
        # Non-negativity constraint for r: r >= 0
        bounds = [(None, None)] * n + [(0, None)]
        
        # Use highs method with aggressive presolve for maximum performance
        result = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs', 
                       options={'presolve': 'on', 'disp': False, 'time_limit': 10.0})
        
        # Extract solution (result.x contains [x1, x2, ..., xn, r])
        solution = result.x[:n]
        
        return {"solution": solution.tolist()}