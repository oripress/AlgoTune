from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List]:
        """
        Solve L0 proximal operator (projection onto L0 ball):
            min_w ||v - w||^2  s.t. ||w||_0 <= k

        This implementation reproduces the reference behavior exactly, including
        stable tie-breaking and Python slicing semantics for k, while running in
        expected O(n) time via selection (np.argpartition).
        """
        v = np.asarray(problem.get("v"))
        k = problem.get("k")

        # Flatten to 1D like the reference
        v = v.ravel()
        n = v.size

        # Number of indices kept equals len(indx[-k:]) from the reference.
        # This implies:
        # - if k > 0: keep min(k, n)
        # - if k == 0: keep n  (because [:] returns the whole array)
        # - if k < 0: keep max(0, n + k)
        if k == 0:
            k_eff = n
        elif k > 0:
            k_eff = k if k < n else n
        else:  # k < 0
            tmp = n + k
            k_eff = tmp if tmp > 0 else 0

        # Trivial cases
        if k_eff <= 0 or n == 0:
            return {"solution": np.zeros(n, dtype=v.dtype).tolist()}
        if k_eff >= n:
            return {"solution": v.tolist()}

        a = np.fabs(v)

        kth = n - k_eff
        # Use indices partition to avoid extra full-array passes later
        part_idx = np.argpartition(a, kth)
        t = a[part_idx[kth]]

        # Candidate top-k indices (unordered)
        cand = part_idx[kth:]

        # Strictly greater-than-threshold indices among candidates (covers all > t)
        strict_idx = cand[a[cand] > t]
        num_strict = strict_idx.size
        m = k_eff - num_strict  # number to take among those equal to t

        if m > 0:
            # Indices equal to the threshold across the whole array
            eq_idx_all = np.flatnonzero(a == t)
            # To match stable mergesort + take last k:
            # among equals, select the last m in original order (largest indices)
            eq_take = eq_idx_all[-m:]
            keep_idx = np.concatenate((strict_idx, eq_take))
        else:
            keep_idx = strict_idx

        pruned = np.zeros(n, dtype=v.dtype)
        pruned[keep_idx] = v[keep_idx]

        return {"solution": pruned.tolist()}
        pruned[keep_idx] = v[keep_idx]

        return {"solution": pruned.tolist()}