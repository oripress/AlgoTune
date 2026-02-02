import numpy as np
from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem, **kwargs):
        T = problem["T"]
        N = problem["N"]
        
        if T == 0:
            return []
        
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]
        
        if N == 0:
            return [-1] * T
        
        # Convert to numpy once
        prices_np = np.asarray(prices, dtype=np.float64)
        probs_np = np.asarray(probs, dtype=np.float64)
        
        # Expected revenue matrix: T x N
        revenue = prices_np * probs_np
        
        # Effective capacities (capped at T)
        caps_np = np.asarray(capacities, dtype=np.int32)
        eff_caps = np.minimum(caps_np, T)
        total_eff_cap = int(eff_caps.sum())
        
        if total_eff_cap == 0:
            return [-1] * T
        
        # Matrix dimension
        n = max(T, total_eff_cap)
        
        # Build cost matrix using repeat
        # Repeat each column of revenue according to capacity
        cost_cols = np.repeat(-revenue, eff_caps, axis=1)
        
        # Pad to make square matrix
        if n > T or n > cost_cols.shape[1]:
            cost = np.zeros((n, n), dtype=np.float64)
            cost[:T, :cost_cols.shape[1]] = cost_cols
        else:
            cost = cost_cols
        
        # Solve assignment problem
        row_idx, col_idx = linear_sum_assignment(cost)
        
        # Build column to product mapping using repeat
        col_to_prod = np.repeat(np.arange(N), eff_caps)
        
        # Build result
        result = [-1] * T
        for r, c in zip(row_idx, col_idx):
            if r < T and c < total_eff_cap:
                result[r] = int(col_to_prod[c])
        
        return result