import numpy as np
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Calculates the outer product of two vectors using broadcasting.
        """
        vec1, vec2 = problem
        return vec1[:, np.newaxis] * vec2