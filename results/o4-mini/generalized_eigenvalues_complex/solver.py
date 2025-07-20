import numpy as np
from scipy.linalg import eigvals as _geig

_solve = np.linalg.solve
_eig = np.linalg.eigvals
_isfinite = np.isfinite
_lexsort = np.lexsort
_LinAlgErr = np.linalg.LinAlgError

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        try:
            # fast path: solve B x = A then standard eigenvalues
            X = _solve(B, A)
            ev = _eig(X)
            if not _isfinite(ev).all():
                raise _LinAlgErr
        except _LinAlgErr:
            # fallback to generalized eigenvalues
            ev = _geig(A, B,
                      left=False, right=False,
                      overwrite_a=True, overwrite_b=True,
                      check_finite=False)
        # sort by real then imag, descending
        idx = _lexsort((-ev.imag, -ev.real))
        return ev[idx].tolist()