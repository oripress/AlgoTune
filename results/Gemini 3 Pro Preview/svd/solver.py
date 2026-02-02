import numpy as np
from scipy.linalg.lapack import dgesdd
from fast_loader import list_to_array

class Solver:
    def solve(self, problem, **kwargs):
        matrix = problem["matrix"]
        
        if isinstance(matrix, np.ndarray):
            A = matrix.astype(np.float64, copy=False)
            if not A.flags['C_CONTIGUOUS']:
                A = np.ascontiguousarray(A)
        else:
            n = len(matrix)
            m = len(matrix[0]) if n > 0 else 0
            if n > 0 and m > 0:
                # list_to_array returns float64
                A = list_to_array(matrix, n, m)
            else:
                A = np.array(matrix, dtype=np.float64)
        
        # We compute SVD of A^T.
        # A is C-contiguous (n, m). A.T is F-contiguous (m, n).
        # LAPACK expects F-contiguous arrays. Passing A.T avoids an internal copy.
        # dgesdd(A.T) returns u_at (V), s, vt_at (U^T)
        
        # compute_uv=1: compute both singular vectors
        # full_matrices=0: reduced SVD
        # overwrite_a=1: allow overwriting the input (A.T view)
        v, s, ut, info = dgesdd(A.T, compute_uv=1, full_matrices=0, overwrite_a=1)
        
        if info > 0:
            raise ValueError("SVD computation did not converge")
            
        # ut is U^T, so U = ut.T
        # v is V
        return {"U": ut.T, "S": s, "V": v}