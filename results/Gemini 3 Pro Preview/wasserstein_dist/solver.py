import numpy as np
try:
    from solver_cython import solve_cython
except ImportError:
    solve_cython = None

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        u = problem["u"]
        v = problem["v"]
        
        # Convert to numpy arrays, ensuring float64 and contiguous
        # We use np.array to create a copy/ensure it's an array
        u_arr = np.ascontiguousarray(u, dtype=np.float64)
        v_arr = np.ascontiguousarray(v, dtype=np.float64)
        
        if solve_cython is not None:
            return solve_cython(u_arr, v_arr)
            
        # Fallback implementation
        u_sum = u_arr.sum()
        v_sum = v_arr.sum()
        
        if u_sum == 0 or v_sum == 0:
            return float(len(u))
            
        u_pmf = u_arr / u_sum
        v_pmf = v_arr / v_sum
        
        u_cdf = np.cumsum(u_pmf)
        v_cdf = np.cumsum(v_pmf)
        
        return np.sum(np.abs(u_cdf[:-1] - v_cdf[:-1]))