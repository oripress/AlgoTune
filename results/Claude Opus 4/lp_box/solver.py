from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def __init__(self):
        # Pre-compile method
        self.method = 'highs-ds'
        
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the LP box problem: minimize c^T x subject to Ax <= b, 0 <= x <= 1
        """
        c = problem["c"]
        A = problem["A"]
        b = problem["b"]
        n = len(c)
        
        # Use list comprehension for bounds (faster than loop)
        bounds = [(0, 1)] * n
        
        # Solve using scipy's linprog with highs-ds method (dual simplex)
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method=self.method)
        
        return {"solution": result.x.tolist()}