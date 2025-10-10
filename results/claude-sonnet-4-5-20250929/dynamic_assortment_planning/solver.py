import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import jit
from typing import Any

@jit(nopython=True, cache=True)
def create_cost_matrix(T, prices, probs, product_indices):
    """Create cost matrix using numba-compiled loops."""
    total_cap = len(product_indices)
    cost = np.empty((T, total_cap), dtype=np.float64)
    for t in range(T):
        for j in range(total_cap):
            prod_idx = product_indices[j]
            cost[t, j] = -prices[prod_idx] * probs[t, prod_idx]
    return cost

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """
        Solve DAP using Hungarian algorithm with numba-optimized cost matrix.
        """
        T = problem["T"]
        N = problem["N"]
        
        if T == 0 or N == 0:
            return [-1] * T
        
        prices = np.array(problem["prices"], dtype=np.float64)
        probs = np.array(problem["probs"], dtype=np.float64)
        capacities = np.array(problem["capacities"], dtype=np.int32)
        
        product_indices = np.repeat(np.arange(N, dtype=np.int32), capacities)
        
        cost = create_cost_matrix(T, prices, probs, product_indices)
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        offer = np.full(T, -1, dtype=np.int32)
        offer[row_ind] = product_indices[col_ind]
        
        return offer.tolist()