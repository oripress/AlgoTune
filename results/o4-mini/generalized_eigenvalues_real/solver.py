import numpy as np
from scipy.linalg import get_lapack_funcs, eigh
from scipy.linalg.blas import get_blas_funcs

# Prepare low-level routines for double precision
_dummy = np.array([[0.]], dtype=np.double)
_potrf, = get_lapack_funcs(('potrf',), (_dummy,))
_trsm, = get_blas_funcs(('trsm',), (_dummy,))

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve A x = λ B x for symmetric A and SPD B.
        Uses Cholesky + BLAS triangular solves + SciPy MRRR eigensolver (evr) for eigenvalues only.
        Returns eigenvalues sorted descending.
        """
        A, B = problem
        # Fortran-order, double precision arrays
        A1 = np.array(A, dtype=np.double, order='F', copy=False)
        B1 = np.array(B, dtype=np.double, order='F', copy=False)
        # Cholesky factorization B = L * L^T (lower), in-place on B1
        B1, info = _potrf(B1, lower=1, overwrite_a=1)
        # Transform A -> L⁻¹ A L⁻ᵀ
        A1 = _trsm(1.0, B1, A1, side=0, lower=1, trans_a=0, diag=0, overwrite_b=1)
        A1 = _trsm(1.0, B1, A1, side=1, lower=1, trans_a=1, diag=0, overwrite_b=1)
        # Compute eigenvalues with MRRR algorithm via SciPy eigh
        w = eigh(A1, lower=True, eigvals_only=True, driver='evr',
                 overwrite_a=True, check_finite=False)
        # Return in descending order
        return w[::-1].tolist()