import numpy as np
from numpy.linalg import eigvals as np_eigvals
from scipy.linalg import eigvals as sp_eigvals

class Solver:
    def solve(self, problem, **kwargs):
        if not isinstance(problem, np.ndarray):
            problem = np.asarray(problem, dtype=np.float64)
        
        n = problem.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [complex(problem[0, 0])]
        if n == 2:
            a00, a01 = float(problem[0, 0]), float(problem[0, 1])
            a10, a11 = float(problem[1, 0]), float(problem[1, 1])
            tr = a00 + a11
            disc = tr * tr - 4.0 * (a00 * a11 - a01 * a10)
            if disc >= 0:
                sq = disc ** 0.5
                return [complex((tr + sq) * 0.5, 0.0), complex((tr - sq) * 0.5, 0.0)]
            sq = (-disc) ** 0.5
            return [complex(tr * 0.5, sq * 0.5), complex(tr * 0.5, -sq * 0.5)]
        
        if n <= 64:
            # For small-medium matrices, scipy eigvals with overwrite is fast
            eigenvalues = sp_eigvals(problem, check_finite=False, overwrite_a=True)
        else:
            # For larger matrices, use scipy too
            if problem.dtype != np.float64:
                problem = problem.astype(np.float64)
            eigenvalues = sp_eigvals(problem, check_finite=False, overwrite_a=True)
        
        # Sort: descending by real part, then by imaginary part
        idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        return eigenvalues[idx].tolist()