import numpy as np
from scipy.linalg import eig as _eig

# Pre-bind numpy routines for speed
_solve = np.linalg.solve
_eig_np = np.linalg.eig
_lexsort = np.lexsort

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        # Convert inputs to float arrays
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        try:
            # For invertible B, compute eigenpairs of inv(B)@A
            C = _solve(B, A)
            evals, evecs = _eig_np(C)
        except np.linalg.LinAlgError:
            # Fallback to generalized eig if B is singular
            evals, evecs = _eig(A, B, overwrite_a=True, overwrite_b=True, check_finite=False)
        # Skip re-normalization (returned vectors are unit‐norm)
        # Sort by descending real part then imaginary part
        idx = _lexsort((-evals.imag, -evals.real))
        # Return as Python lists
        return evals[idx].tolist(), evecs[:, idx].T.tolist()