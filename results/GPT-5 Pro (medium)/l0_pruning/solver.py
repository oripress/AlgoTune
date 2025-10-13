from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Solve the L0 proximal operator (projection onto L0-ball) efficiently.

        We match the reference implementation exactly (including its tie-breaking via
        stable mergesort and its behavior for edge cases like k == 0), but we use an
        O(n) approach with np.partition for the common case 0 < k < n.

        :param problem: A dictionary with keys:
                        - "v": list of real numbers
                        - "k": integer hyperparameter
        :return: A dictionary with key "solution": list of numbers
        """
        v = np.array(problem.get("v"))
        # Ensure 1D
        v = v.flatten()
        n = v.size

        # Handle trivial empty case
        if n == 0:
            return {"solution": []}

        k = problem.get("k")
        # Cast k to int for indexing, consistent with numpy slicing behavior
        try:
            k = int(k)
        except Exception:
            # Fallback; though inputs should be valid
            k = int(k)

        # Reference behavior quirks to match exactly:
        # - If k == 0: due to Python slicing [-0:] == [0:], the reference keeps all entries.
        # - If k >= n: taking the last k entries yields all entries.
        if k == 0 or k >= n:
            return {"solution": v.tolist()}

        pruned = np.zeros_like(v)

        # For negative k, replicate the reference's mergesort-based behavior exactly.
        if k < 0:
            idx = np.argsort(np.abs(v), kind="mergesort")
            remaining_idx = idx[-k:]  # when -k > n, this yields empty slice, as in reference
            if remaining_idx.size:
                pruned[remaining_idx] = v[remaining_idx]
            return {"solution": pruned.tolist()}

        # Common fast path: 0 < k < n
        abs_v = np.abs(v)

        # Use partition to find the cutoff t such that exactly (n - k) elements are <= t's position
        # t is the (n - k)-th smallest element (0-based indexing)
        t = np.partition(abs_v, n - k)[n - k]

        # Count how many are strictly greater than the threshold
        mask_gt = abs_v > t
        num_gt = int(mask_gt.sum())

        # Select indices with abs > t
        if num_gt > 0:
            pruned[mask_gt] = v[mask_gt]

        # Add remaining needed from those equal to t, choosing the largest original indices
        # to match stable mergesort's tie-breaking when taking the last k elements.
        remaining = k - num_gt
        if remaining > 0:
            # Indices where abs == t
            eq_mask = ~mask_gt & (abs_v == t)
            if remaining >= int(eq_mask.sum()):
                # All equal-to-threshold indices are needed
                if eq_mask.any():
                    pruned[eq_mask] = v[eq_mask]
            else:
                # Choose the last 'remaining' equal-to-threshold indices (highest original indices)
                eq_idx = np.flatnonzero(eq_mask)
                if remaining > 0:
                    chosen_eq = eq_idx[-remaining:]
                    pruned[chosen_eq] = v[chosen_eq]

        return {"solution": pruned.tolist()}