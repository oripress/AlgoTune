import numpy as np
from scipy.linalg.lapack import dgetrf

try:
    import solver_utils
except ImportError:
    solver_utils = None


class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs):
        A = problem["matrix"]
        # Ensure A is a numpy array and Fortran contiguous for LAPACK
        A = np.asfortranarray(A, dtype=np.float64)
        n = A.shape[0]
        
        # dgetrf overwrites A with LU
        lu_packed, piv, info = dgetrf(A, overwrite_a=True)
        
        if info < 0:
            raise ValueError(f"Illegal value in argument {-info} of dgetrf")
        
        if solver_utils:
            # Use Cython implementation
            # piv is 0-based indices
            # lu_packed is F-contiguous
            # We need to ensure piv is int32 for Cython
            piv = piv.astype(np.int32)
            results = solver_utils.extract_results(lu_packed, piv, n)
            return {"LU": results}
        else:
            # Fallback to python implementation
            # Extract L and U
            L = np.tril(lu_packed, k=-1)
            np.fill_diagonal(L, 1.0)
            U = np.triu(lu_packed)
            
            # Construct P
            p = np.arange(n)
            for i in range(n):
                target = piv[i]
                if target != i:
                    p[i], p[target] = p[target], p[i]
            
            P = np.zeros((n, n))
            P[p, np.arange(n)] = 1.0
            
            solution = {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
            return solution