import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the water-filling problem using a highly optimized O(n log n)
        analytical algorithm. This version is tuned for performance by using
        np.sort, an efficient early-terminating loop, and in-place NumPy
        operations to minimize memory allocations and overhead.
        """
        alpha = np.asarray(problem["alpha"], dtype=np.float64)
        P_total = float(problem["P_total"])
        n = alpha.size

        if n == 0:
            return {"x": [], "Capacity": 0.0}

        # 1. Sort noise levels. np.sort is often slightly faster than argsort
        # followed by fancy indexing. It returns a sorted copy. O(n log n)
        a_sorted = np.sort(alpha)

        # 2. Compute prefix sums of sorted noise levels. O(n)
        prefix_sum_a = np.cumsum(a_sorted)

        # 3. Find the correct water level 'w' using an early-terminating loop.
        # This is more efficient than non-short-circuiting vectorized approaches.
        w = 0.0
        for k in range(1, n + 1):
            w_candidate = (P_total + prefix_sum_a[k - 1]) / k
            if k == n or w_candidate <= a_sorted[k]:
                w = w_candidate
                break
        
        # 4. Calculate the optimal power allocation x using in-place operations
        # to reduce memory overhead.
        x = w - alpha
        np.maximum(x, 0.0, out=x)

        # 5. Rescale in-place to ensure the power constraint is met exactly.
        current_sum = np.sum(x)
        if current_sum > 1e-9:
            x *= (P_total / current_sum)
        
        # 6. Calculate the final capacity using memory-optimized in-place operations.
        # This avoids creating a second temporary array for the log operation.
        temp = alpha + x
        np.log(temp, out=temp)
        capacity = np.sum(temp)

        return {"x": x.tolist(), "Capacity": float(capacity)}