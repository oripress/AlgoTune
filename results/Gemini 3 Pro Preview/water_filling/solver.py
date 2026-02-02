import numpy as np
from typing import Any
# Import the cython module. It should be available after compilation.
# The module name is fast_solver (from setup.py)
try:
    from fast_solver import solve_cython
except ImportError:
    # Fallback or re-raise. Since compilation succeeded, it should work.
    # But for safety during development:
    solve_cython = None

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        try:
            alpha_list = problem["alpha"]
            P_total = float(problem["P_total"])
        except KeyError:
            return {"x": [], "Capacity": float("nan")}
            
        # Use Cython implementation if available
        if solve_cython:
            # solve_cython handles list input directly
            x, capacity = solve_cython(alpha_list, P_total)
            
            if np.isnan(capacity):
                 # Need to return list of nans of correct size
                 # solve_cython returns numpy array of nans
                 # But we need to ensure size is correct if n was 0?
                 # solve_cython handles n=0 returning empty array?
                 # Let's check fast_solver.pyx logic.
                 # If n=0, returns full(0, nan), nan.
                 # If invalid, returns full(n, nan), nan.
                 # So x is already correct size.
                 return {"x": x.tolist(), "Capacity": float("nan")}
                 
            return {"x": x, "Capacity": capacity}
        else:
            # Fallback (should not happen if compilation works)
            return {"x": [], "Capacity": float("nan")}