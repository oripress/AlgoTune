import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

class Solver:
    def __init__(self):
        # LAPACK driver for generalized DC eigenproblem
        self._sygvd = None

    def solve(self, problem, **kwargs):
        """
        Solve A x = lambda B x using LAPACK divide-and-conquer (dsygvd).
        Returns eigenvalues in descending order and B-orthonormal eigenvectors.
        """
        A, B = problem
        # Prepare float64 Fortran-ordered arrays
        A = np.array(A, dtype=np.float64, order='F', copy=False)
        B = np.array(B, dtype=np.float64, order='F', copy=False)
        # Lazy-load dsygvd
        if self._sygvd is None:
            self._sygvd, = get_lapack_funcs(('sygvd',), (A, B))
        # Compute eigenvalues and eigenvectors: itype=1, jobz='V', uplo='L'
        w, v, info = self._sygvd(A, B,
                                  itype=1, jobz='V', uplo='L',
                                  overwrite_a=1, overwrite_b=1)
        if info != 0:
            raise np.linalg.LinAlgError(f"LAPACK dsygvd failed to converge (info={info})")
        # Sort descending
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]
        # Convert to Python lists
        return w.tolist(), [v[:, i].tolist() for i in range(v.shape[1])]