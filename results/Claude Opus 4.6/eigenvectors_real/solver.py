import numpy as np
from scipy.linalg import lapack
import math

_dsyevd = lapack.dsyevd

class Solver:
    def solve(self, problem, **kwargs):
        """Compute eigenvalues and eigenvectors of a real symmetric matrix."""
        if isinstance(problem, np.ndarray):
            if problem.dtype == np.float64:
                A = problem
            else:
                A = problem.astype(np.float64)
        else:
            A = np.array(problem, dtype=np.float64)
        
        n = A.shape[0]
        
        if n == 0:
            return ([], [])
        
        if n == 1:
            return ([float(A[0, 0])], [[1.0]])
        
        if n == 2:
            a = float(A[0, 0])
            b = float(A[0, 1])
            d = float(A[1, 1])
            trace = a + d
            diff = a - d
            disc = math.sqrt(diff * diff + 4.0 * b * b)
            l1 = (trace + disc) * 0.5
            l2 = (trace - disc) * 0.5
            
            if abs(b) > 1e-15:
                v1x = b
                v1y = l1 - a
                inv_norm = 1.0 / math.sqrt(v1x * v1x + v1y * v1y)
                v1x *= inv_norm
                v1y *= inv_norm
                return ([l1, l2], [[v1x, v1y], [-v1y, v1x]])
            else:
                if a >= d:
                    return ([l1, l2], [[1.0, 0.0], [0.0, 1.0]])
                else:
                    return ([l1, l2], [[0.0, 1.0], [1.0, 0.0]])
        
        # For symmetric matrices: A^T = A, so C-contiguous A.T is F-contiguous
        # and represents the same matrix. Use overwrite_a=0 if A shares memory.
        if A.flags['C_CONTIGUOUS']:
            # A.T is F-contiguous (view of same data) - symmetric means same matrix
            # Need to check if A is writeable and if we can overwrite it
            if A.flags['WRITEABLE'] and not np.shares_memory(A, problem if isinstance(problem, np.ndarray) else A):
                A_f = A.T  # F-contiguous view, no copy
                w, v, info = _dsyevd(A_f, lower=1, overwrite_a=1, compute_v=1)
            else:
                A_f = np.asfortranarray(A)
                w, v, info = _dsyevd(A_f, lower=1, overwrite_a=1, compute_v=1)
        else:
            A_f = np.asfortranarray(A)
            w, v, info = _dsyevd(A_f, lower=1, overwrite_a=1, compute_v=1)
        
        result_vecs = np.ascontiguousarray(v[:, ::-1].T)
        return (w[::-1].tolist(), result_vecs.tolist())