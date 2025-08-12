import numpy as np
from scipy.linalg import get_lapack_funcs

# Prefetch LAPACK driver for generalized eigensolver (dsygvd) to minimize per-call overhead
_sygvd, = get_lapack_funcs(
    ('sygvd',),
    (np.zeros((1, 1), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)),
)

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the generalized eigenproblem A x = λ B x for symmetric A and SPD B.
        Uses LAPACK's dsygvd divide-and-conquer solver via SciPy for maximum C-side performance.
        Returns eigenvalues in descending order and B-orthonormal eigenvectors.
        """
        A_in, B_in = problem
        # Ensure float64 arrays, Fortran-order for LAPACK
        A = np.array(A_in, dtype=np.float64, order='F')
        B = np.array(B_in, dtype=np.float64, order='F')
        # Compute eigenvalues and eigenvectors: A v = w B v
        res = _sygvd(
            A, B,
            itype=1,    # A x = λ B x
            jobz='V',   # compute eigenvectors
            uplo='L',   # use lower triangle
            overwrite_a=True,
            overwrite_b=True,
        )
        # res = (A_out, B_out, w, v, info)
        w = res[-3]
        v = res[-2]
        info = res[-1]
        if info != 0:
            raise RuntimeError(f'LAPACK dsygvd failed with info={info}')
        # Reverse for descending eigenvalues
        w = w[::-1]
        v = v[:, ::-1]
        # Return Python lists
        return w.tolist(), v.T.tolist()