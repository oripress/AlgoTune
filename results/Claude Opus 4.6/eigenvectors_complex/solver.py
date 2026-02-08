import numpy as np
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        w, v = np.linalg.eig(np.asarray(problem, dtype=np.float64))
        idx = np.lexsort((-w.imag, -w.real))
        # np.linalg.eig already returns unit-norm eigenvectors
        return v[:, idx].T.tolist()