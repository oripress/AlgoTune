from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """Solve the linear system Ax = b."""
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        x = np.linalg.solve(A, b)
        return x.tolist()