import numpy as np
from typing import Any, Dict, List

class Solver:
    """
    Project vector v onto the L0 ball of size k:
        min_w ||v - w||^2  s.t. ||w||_0 <= k

    Efficient implementation:
    - Uses numpy.partition for average O(n) threshold selection.
    - Falls back to stable np.argsort(..., kind='mergesort') when NaNs are present
      or when partitioning isn't appropriate.
    - Tie-handling: when multiple entries equal the threshold, selects the later
      original indices (matching a stable mergesort then taking the last k indices).
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        v = problem.get("v", [])
        k = problem.get("k", 0)

        # Convert v to 1-D numpy array of floats
        v_arr = np.asarray(v, dtype=float).ravel()
        n = v_arr.size

        # Robust conversion of k to integer
        try:
            k = int(k)
        except Exception:
            try:
                k = int(float(k))
            except Exception:
                k = 0

        # Trivial cases
        if n == 0:
            return {"solution": []}
        if k <= 0:
            return {"solution": np.zeros(n, dtype=float).tolist()}
        if k >= n:
            return {"solution": v_arr.tolist()}

        abs_v = np.abs(v_arr)

        # If NaNs present, use stable mergesort selection (matches reference behavior)
        if np.isnan(abs_v).any():
            idx = np.argsort(abs_v, kind="mergesort")
            chosen = idx[-k:]
            w = np.zeros_like(v_arr)
            w[chosen] = v_arr[chosen]
            return {"solution": w.tolist()}

        # Find the k-th largest magnitude threshold (position n-k)
        pos = n - k
        try:
            T = np.partition(abs_v, pos)[pos]
        except Exception:
            # Fallback to stable mergesort if partition fails
            idx = np.argsort(abs_v, kind="mergesort")
            chosen = idx[-k:]
            w = np.zeros_like(v_arr)
            w[chosen] = v_arr[chosen]
            return {"solution": w.tolist()}

        # Select entries strictly greater than T
        greater_mask = abs_v > T
        cnt_greater = int(np.count_nonzero(greater_mask))
        remaining = k - cnt_greater

        selected = greater_mask.copy()

        if remaining > 0:
            # Indices equal to threshold; pick the last `remaining` ones (later original indices)
            equal_idx = np.flatnonzero(abs_v == T)
            if equal_idx.size == 0:
                # Unexpected; fallback to stable mergesort
                idx = np.argsort(abs_v, kind="mergesort")
                chosen = idx[-k:]
                w = np.zeros_like(v_arr)
                w[chosen] = v_arr[chosen]
                return {"solution": w.tolist()}
            if remaining >= equal_idx.size:
                selected[equal_idx] = True
            else:
                selected[equal_idx[-remaining:]] = True

        w = np.zeros_like(v_arr)
        w[selected] = v_arr[selected]
        return {"solution": w.tolist()}