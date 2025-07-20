import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solves the L0 proximal operator problem.
        min_w ||v-w||^2 s.t. ||w||_0 <= k
        """
        v_list = problem.get("v")
        k = problem.get("k")

        v = np.array(v_list, dtype=np.float64).flatten()
        
        n = v.size
        
        if k <= 0:
            # If k is 0 or less, all elements must be zero.
            return {"solution": [0.0] * n}
        if k >= n:
            # If k is n or more, all elements are kept.
            return {"solution": v.tolist()}

        # Use argpartition to find the indices of the k largest elements in O(n).
        # This is much faster than a full sort (O(n log n)).
        # We find the (n-k)th smallest element in terms of absolute value.
        # The elements from this partition point to the end of the array
        # will be the k largest elements.
        indices_to_keep = np.argpartition(np.abs(v), n - k)[n - k:]

        w = np.zeros_like(v)
        w[indices_to_keep] = v[indices_to_keep]

        return {"solution": w.tolist()}