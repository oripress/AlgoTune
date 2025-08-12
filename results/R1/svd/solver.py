import numpy as np
import os
from scipy.linalg.lapack import get_lapack_funcs

# Configure MKL for best performance
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_DYNAMIC"] = "FALSE"

class Solver:
    def solve(self, problem, **kwargs):
        matrix = problem["matrix"]
        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0
        
        # Convert to float64 Fortran-ordered array for efficiency
        A = np.array(matrix, dtype=np.float64, order='F')
        
        # Get the LAPACK gesdd function
        gesdd, = get_lapack_funcs(('gesdd',), (A,))
        
        # Compute SVD using low-level LAPACK call
        u, s, vt, info = gesdd(A, full_matrices=False, compute_uv=True, overwrite_a=True)
        
        # Check for successful computation
        if info < 0:
            raise ValueError(f"Illegal value in argument {-info} of internal gesdd")
        elif info > 0:
            raise ValueError("SVD computation did not converge")
        
        return {
            "U": u,
            "S": s,
            "V": vt.T
        }