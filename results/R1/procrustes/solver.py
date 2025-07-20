import numpy as np
from scipy.linalg import get_lapack_funcs

class Solver:
    def solve(self, problem, **kwargs):
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}
        
        try:
            # Convert to arrays with optimal memory layout
            A_arr = np.array(A, dtype=np.float64, order='C')
            B_arr = np.array(B, dtype=np.float64, order='C')
        except Exception:
            return {}
        
        if A_arr.shape != B_arr.shape:
            return {}
        n = A_arr.shape[0]
        if A_arr.ndim != 2 or n != A_arr.shape[1]:
            return {}
        
        # Handle 1x1 matrix separately
        if n == 1:
            return {"solution": [[1.0]] if B_arr[0,0] * A_arr[0,0] >= 0 else [[-1.0]]}
        
        # Handle 2x2 matrix with direct formula
        if n == 2:
            M = B_arr @ A_arr.T
            a, b, c, d = M.flat
            det = a*d - b*c
            if det >= 0:
                return {"solution": [[d, -b], [-c, a]] * (1/np.sqrt(det))}
            else:
                return {"solution": [[-d, b], [c, -a]] * (1/np.sqrt(-det))}
        
        # Compute M = B A^T
        M = B_arr @ A_arr.T
        
        # Use low-level LAPACK SVD (gesdd is typically fastest)
        try:
            # Get LAPACK function
            gesdd, = get_lapack_funcs(['gesdd'], (M,))
            
            # Compute SVD with LAPACK
            u, s, vt, info = gesdd(M, full_matrices=0, overwrite_a=1)
            
            if info == 0:
                # Success - compute G = u @ vt
                G = u @ vt
                return {"solution": G.tolist()}
        except Exception:
            pass
        
        # Fallback to standard SVD if LAPACK fails
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        G = U @ Vt
        return {"solution": G.tolist()}