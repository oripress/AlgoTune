# Efficient linear system solver using NumPy
from typing import Any, List
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the linear system Ax = b.

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" (square matrix) and "b" (right‑hand side vector).

        Returns
        -------
        List[float]
            Solution vector x such that A @ x = b.
        """
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        x = np.linalg.solve(A, b)
        return x.tolist()