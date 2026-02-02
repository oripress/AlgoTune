import numpy as np
try:
    from fast_solver import solve_cython
except ImportError:
    solve_cython = None

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=float)
        
        if solve_cython:
            # Ensure A is contiguous and float64
            if not A.flags.c_contiguous or A.dtype != np.float64:
                A = np.ascontiguousarray(A, dtype=np.float64)
            return solve_cython(A)
        else:
            # Fallback to pure python implementation if import fails
            n = A.shape[0]
            P = np.eye(n)
            curr_A = A.copy()
            
            for _ in range(20):
                norm_A = np.linalg.norm(curr_A, np.inf)
                
                if norm_A < 1e-10:
                    P = (P + P.T) / 2.0
                    return {"is_stable": True, "P": P}
                
                if norm_A > 1e10:
                    return {"is_stable": False, "P": None}
                
                M = P @ curr_A
                P += curr_A.T @ M
                curr_A = curr_A @ curr_A
                
            return {"is_stable": False, "P": None}