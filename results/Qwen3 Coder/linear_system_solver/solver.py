import numpy as np
import scipy.linalg
from typing import Any
class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Solve the linear system Ax = b.
        
        Args:
            problem: A dictionary with keys "A" and "b"
            
        Returns:
            A list of numbers representing the solution vector x.
        """
        A = np.array(problem["A"], dtype=np.float64, order='C')
        b = np.array(problem["b"], dtype=np.float64, order='C')
        x = np.linalg.solve(A, b)
        return x.tolist()