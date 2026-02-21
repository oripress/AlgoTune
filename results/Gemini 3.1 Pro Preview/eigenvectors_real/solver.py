import numpy as np
import scipy.linalg
from typing import Any

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        eigenvalues, eigenvectors = scipy.linalg.eigh(problem, driver='evd')
        return eigenvalues[::-1].tolist(), eigenvectors.T[::-1].tolist()