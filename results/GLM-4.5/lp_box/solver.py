from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp box problem using scipy.optimize.linprog with 'highs' solver
        and optimized numpy tile bounds for maximum performance.

        :param problem: A dictionary of the lp box problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp box problem.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Convert to numpy arrays with optimal dtype
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        n = len(c)
        
        # Use numpy tile for bounds creation - potentially more efficient
        bounds = np.tile([0, 1], (n, 1))
        
        # Solve using highs with minimal overhead
        result = linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method='highs'
        )
        
        return {"solution": result.x.tolist()}
        
        # Solve
        h.run()
        
        # Get solution
        solution = h.getSolution().col_value
        
        return {"solution": solution}