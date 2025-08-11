import numpy as np

# Local binding to reduce attribute lookup overhead
_cholesky = np.linalg.cholesky

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the Cholesky factorization A = L @ L.T using NumPy's LAPACK-backed routine.
        This implementation keeps Python-side overhead minimal by calling the bound function directly.
        """
        A = problem["matrix"]
        L = _cholesky(A)
        return {"Cholesky": {"L": L}}