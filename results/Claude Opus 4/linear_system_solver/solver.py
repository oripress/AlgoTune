from typing import Any
import numpy as np
from numba import njit

@njit
def solve_linear_system(A, b):
    """JIT-compiled linear system solver."""
    return np.linalg.solve(A, b)

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Solve the linear system Ax = b using Numba JIT-compiled solver.
        
        Args:
            problem (dict): A dictionary with keys "A" and "b".
        
        Returns:
            list: A list of numbers representing the solution vector x.
        """
        # Convert to numpy arrays
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        # Use JIT-compiled solver
        x = solve_linear_system(A, b)
        
        # Convert back to Python list
        return x.tolist()