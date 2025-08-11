import math
import numpy as np
from typing import List, Any
from numba import njit

@njit
def _btsp_dp(w):
    """
    Dynamic programming over subsets for the Bottleneck TSP.
    Returns:
        dp     – (mask, last) → minimal possible bottleneck for a path
                  that starts at city 0, visits exactly the cities in `mask`,
                  and ends at `last`.
        parent – predecessor city for each state (used for reconstruction).
    """
    n = w.shape[0]
    size = 1 << n
    INF = 1e18

    dp = np.full((size, n), INF, dtype=np.float64)
    parent = np.full((size, n), -1, dtype=np.int64)

    start_mask = 1          # only city 0 visited
    dp[start_mask, 0] = 0.0

    full_mask = (1 << n) - 1

    for mask in range(size):
        # we only care about masks that contain the start city
        if (mask & start_mask) == 0:
            continue
        for u in range(n):
            if (mask & (1 << u)) == 0:
                continue
            cur = dp[mask, u]
            if cur >= INF:
                continue
            for v in range(n):
                if (mask & (1 << v)) != 0:
                    continue
                new_mask = mask | (1 << v)
                # bottleneck of the new path = max(previous bottleneck, edge u‑v)
                new_bottleneck = cur if cur > w[u, v] else w[u, v]
                if new_bottleneck < dp[new_mask, v]:
                    dp[new_mask, v] = new_bottleneck
                    parent[new_mask, v] = u
    return dp, parent

class Solver:
    def solve(self, problem: List[List[float]], **kwargs) -> List[int]:
        """
        Solve the Bottleneck Traveling Salesman Problem (BTSP) using a
        DP over subsets accelerated with Numba. Returns a Hamiltonian
        cycle starting and ending at city 0.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Convert to NumPy array for Numba compatibility
        w = np.array(problem, dtype=np.float64)

        dp, parent = _btsp_dp(w)

        full_mask = (1 << n) - 1
        INF = math.inf
        best_bottleneck = INF
        last = -1
        for u in range(1, n):
            bott = max(dp[full_mask, u], w[u, 0])
            if bott < best_bottleneck:
                best_bottleneck = bott
                last = u

        if last == -1:
            return [0, 0]

        # Reconstruct path backwards from last to start (0)
        path_rev = []
        mask = full_mask
        v = last
        while mask != 1:  # while mask != start_mask
            path_rev.append(int(v))
            prev = parent[mask, v]
            mask ^= (1 << v)
            v = int(prev)
        path_rev.append(0)  # start city

        # Reverse to get forward order and close the tour
        path = [int(x) for x in reversed(path_rev)]
        path.append(0)
        return path