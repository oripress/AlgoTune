from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Solve the linear system Ax = b using highly optimized NumPy solver.

        Args:
            problem (dict): A dictionary with keys "A" and "b".

        Returns:
            list: A list of numbers representing the solution vector x.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Pre-allocate arrays and use asarray to avoid unnecessary copies
        A = np.asarray(problem["A"], dtype=np.float64, order='C')
        b = np.asarray(problem["b"], dtype=np.float64, order='C')
        
        # Use numpy's solve with optimized settings
        x = np.linalg.solve(A, b)
        
        # Convert to list efficiently
        return x.tolist()