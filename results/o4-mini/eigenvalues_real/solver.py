import numpy as np
from scipy.linalg import eigvalsh

class Solver:
    def solve(self, problem, **kwargs):
        # Prepare Fortran-ordered array to eliminate extra copies
        a = np.array(problem, dtype=np.float64, order='F')
        # Compute eigenvalues-only via MRRR driver, in-place, skip checks
        w = eigvalsh(a, driver='evr', overwrite_a=True, check_finite=False)
        # Return in descending order
        return w[::-1].tolist()
        return w[::-1].tolist()
        return w[::-1].tolist()
        a = np.array(problem, dtype=np.float64, order='F')
        w = _dsyevr(a, 'N', 'A', 'U', 0.0, 0.0, 0, 0, 0.0)[0]
        return w[::-1].tolist()