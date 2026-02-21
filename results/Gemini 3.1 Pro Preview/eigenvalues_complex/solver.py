import numpy as np
import scipy.linalg
from typing import Any

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        eigenvalues = scipy.linalg.eigvals(problem, overwrite_a=True, check_finite=False)
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues.tolist()