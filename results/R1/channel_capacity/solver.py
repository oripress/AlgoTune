import math
from typing import Any
import cvxpy as cp
import numpy as np
from scipy.special import xlogy

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        # Extract and validate input matrix
        P = problem.get("P")
        if P is None:
            return None
        P_arr = np.array(P, dtype=np.float64)
        if P_arr.ndim != 2:
            return None
        m, n = P_arr.shape
        if n == 0 or m == 0:
            return None
        
        # Validate column sums
        col_sums = np.sum(P_arr, axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6):
            return None
        
        # Precompute constant vector c accurately
        ln2 = math.log(2)
        c_vec = np.sum(xlogy(P_arr, P_arr), axis=0) / ln2

        # Optimization variables and constraints
        x = cp.Variable(n)
        constraints = [cp.sum(x) == 1, x >= 0]
        y = P_arr @ x  # Output distribution
        
        # Mutual information objective using cp.entr
        mutual_information = c_vec @ x + cp.sum(cp.entr(y)) / ln2
        problem = cp.Problem(cp.Maximize(mutual_information), constraints)
        
        # Solve with ECOS first, fallback to SCS if needed
        try:
            problem.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, max_iters=10000, verbose=False)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Try SCS if ECOS didn't converge
                problem.solve(solver=cp.SCS, eps=1e-6, max_iters=5000, verbose=False)
        except (cp.SolverError, Exception):
            return None

        # Validate solution
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if problem.value is None or x.value is None:
            return None
            
        # Ensure solution meets constraints
        x_val = np.maximum(x.value, 0)
        total = np.sum(x_val)
        if total <= 1e-10:  # Avoid division by zero
            return None
        x_val /= total  # Renormalize
        
        # Calculate mutual information for the valid solution
        y_val = P_arr @ x_val
        # Compute mutual information directly
        non_zero = y_val > 1e-15
        H_Y = -np.sum(xlogy(y_val[non_zero], y_val[non_zero])) / ln2
        H_Y_given_X = -np.sum(x_val * c_vec)
        mutual_info_val = H_Y - H_Y_given_X
        
        return {"x": x_val.tolist(), "C": mutual_info_val}