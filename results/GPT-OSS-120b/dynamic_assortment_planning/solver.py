from typing import Any, List
import numpy as np
from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Solve the Dynamic Assortment Planning problem using a fast
        Hungarian (linear sum assignment) algorithm.

        We create a copy for each unit of product capacity and match each
        period to either a product copy (earning expected revenue) or to a
        dummy copy representing idling (zero revenue). The assignment that
        maximizes total expected revenue is obtained by minimizing the
        negative revenue matrix.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        # Build list mapping each copy index to its product
        copy_to_product = []
        for i, cap in enumerate(capacities):
            copy_to_product.extend([i] * cap)
        total_copies = len(copy_to_product)

        # Add dummy copies for idle option (one per period is sufficient)
        dummy_copies = T
        num_cols = total_copies + dummy_copies

        # Cost matrix: we minimize cost, so use negative expected revenue
        cost = np.zeros((T, num_cols), dtype=float)

        # Fill real product copies
        for col, prod in enumerate(copy_to_product):
            for t in range(T):
                cost[t, col] = -prices[prod] * probs[t][prod]

        # Dummy columns already zero (revenue 0)

        # Solve assignment (minimize total cost)
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build schedule: -1 for idle, otherwise product index
        schedule = [-1] * T
        for r, c in zip(row_ind, col_ind):
            if c < total_copies:
                schedule[r] = copy_to_product[c]
            else:
                schedule[r] = -1
        return schedule