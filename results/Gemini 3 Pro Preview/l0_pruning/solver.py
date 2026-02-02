import numpy as np
try:
    from solver_cython import solve_cython
except ImportError:
    # Fallback or re-raise if compilation failed
    raise

class Solver:
    def solve(self, problem, **kwargs):
        v_list = problem.get("v")
        k = problem.get("k")
        
        # Convert to array. 
        v = np.array(v_list, dtype=np.float64)
        v = v.ravel()
        n = v.size
        
        if k == 0:
            return {"solution": [0.0] * n}
            
        if k >= n:
            return {"solution": v.tolist()}
            
        # Call Cython implementation
        # solve_cython modifies v in-place and returns it (as a memoryview/array)
        solve_cython(v, k)
        
        return {"solution": v}