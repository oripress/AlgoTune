import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        """
        Solve the linear system Ax = b using NumPy's optimized solver with overwrite flags.
        """
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        x = np.linalg.solve(A, b)
        return x.tolist()