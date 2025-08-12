from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp box problem using scipy.optimize.linprog.
        
        :param problem: A dictionary of the lp box problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp box problem.
        """
        c = problem["c"]
        A_ub = problem["A"]
        b_ub = problem["b"]
        n = len(c)
        
        # Bounds for variables (0 <= x <= 1)
        bounds = [(0.0, 1.0)] * n
        
        # Solve using scipy's linprog with optimized settings
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        return {"solution": result.x.tolist()}