import numpy as np
from scipy.linalg.lapack import dsyevr, ssyevr
from typing import Any

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        if problem.dtype == np.float32:
            w, z, m, isuppz, info = ssyevr(problem, compute_v=0, overwrite_a=1)
        else:
            w, z, m, isuppz, info = dsyevr(problem, compute_v=0, overwrite_a=1)
        return w[::-1].tolist()