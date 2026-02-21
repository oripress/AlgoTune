import numpy as np
import ot
from typing import Any

from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[int]:
        T = problem["T"]
        N = problem["N"]
        
        prices = np.fromiter(problem["prices"], dtype=np.float64, count=N)
        capacities = np.fromiter((c if c < T else T for c in problem["capacities"]), dtype=np.int32, count=N)
        
        cost = np.array(problem["probs"], dtype=np.float64)
        cost *= -prices
        
        total_cap = int(capacities.sum())
        
        if total_cap == 0:
            return [-1] * T
            
        # Expand products based on capacities
        expanded_cost = np.repeat(cost, capacities, axis=1)
        product_indices = np.repeat(np.arange(N, dtype=np.int32), capacities)
        
        row_ind, col_ind = linear_sum_assignment(expanded_cost)
        
        offer = [-1] * T
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if expanded_cost[r, c] < 0:
                offer[r] = int(product_indices[c])
                
        return offer