import numpy as np
try:
    from scipy.linalg.lapack import dsygvx
except ImportError:
    dsygvx = None
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric A and SPD B.
        Uses LAPACK's dsygvx for eigenvalues-only via MRRR algorithm when available,
        falling back to SciPy's eigh with generalized divide-and-conquer.
        Returns eigenvalues sorted in descending order.
        """
        A, B = problem
        # Ensure float64 and Fortran order for LAPACK performance
        A = np.array(A, dtype=np.float64, order='F', copy=False)
        B = np.array(B, dtype=np.float64, order='F', copy=False)
        # Try LAPACK dsygvx for eigenvalues-only general problem
        if dsygvx is not None:
            try:
                res = dsygvx(1, 'N', 'A', 'L', A, B, overwrite_a=1, overwrite_b=1)
                # Extract eigenvalues vector from results
                w = None
                for arr in res:
                    if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.shape[0] == A.shape[0]:
                        w = arr
                        break
                if w is not None:
                    return w[::-1].tolist()
            except Exception:
                pass
        # Fallback: SciPy generalized divide-and-conquer via eigh
        try:
            w = eigh(A, B, eigvals_only=True, check_finite=False,
                     driver='gvd', type=1, overwrite_a=True, overwrite_b=True)
        except Exception:
            w = eigh(A, B, eigvals_only=True, check_finite=False)
        # Return descending order
        return w[::-1].tolist()