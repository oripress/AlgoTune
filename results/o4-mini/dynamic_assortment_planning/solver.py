from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem: dict, **kwargs) -> List[int]:
        # parse inputs
        T = problem["T"]
        N = problem["N"]
        prices = np.array(problem["prices"], dtype=float)
        capacities = np.array(problem["capacities"], dtype=int)
        probs = np.array(problem["probs"], dtype=float)  # shape (T, N)

        # compute expected revenue weights
        weight = probs * prices  # shape (T, N)
        total_cap = int(capacities.sum())
        if total_cap == 0:
            return [-1] * T

        # build product slot indices
        slots = np.repeat(np.arange(N, dtype=int), capacities)
        # cost matrix (negative for maximization)
        cost = -weight[:, slots]  # shape (T, total_cap)

        # solve rectangular assignment (minimize cost => maximize revenue)
        row_ind, col_ind = linear_sum_assignment(cost)

        # build solution with idle default (-1)
        offer = np.full(T, -1, dtype=int)
        offer[row_ind] = slots[col_ind]
        return offer.tolist()