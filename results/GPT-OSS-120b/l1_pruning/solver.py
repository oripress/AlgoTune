import numpy as np
from typing import Any, Dict, List

def _project_l1(v: np.ndarray, k: float) -> np.ndarray:
    """
    Projection of vector v onto the L1 ball of radius k.
    Exact O(n log n) algorithm (sorting + linear scan) using only NumPy.
    """
    # Edge cases
    if k <= 0.0:
        return np.zeros_like(v)

    abs_v = np.abs(v)
    if np.sum(abs_v) <= k:
        return v.copy()

    # Sort absolute values in descending order
    u = np.sort(abs_v)[::-1]

    # Cumulative sum of sorted values
    cssv = np.cumsum(u)

    # Compute thresholds and find the largest index where u_i > (cssv_i - k) / (i+1)
    # This follows the standard O(n log n) algorithm for L1 ball projection.
    # t_i = (cssv_i - k) / (i+1)
    t = (cssv - k) / (np.arange(1, len(u) + 1))
    # Find indices where u > t
    idx = np.where(u > t)[0]
    if idx.size == 0:
        # No positive entries satisfy the condition; set theta to zero
        theta = 0.0
    else:
        rho = idx[-1]  # last index satisfying the condition
        theta = (cssv[rho] - k) / (rho + 1)
    # Soft‑thresholding
    w = np.sign(v) * np.maximum(abs_v - theta, 0.0)
    return w

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Project vector v onto the L1 ball of radius k.

        Parameters
        ----------
        problem : dict
            - "v": list of floats, the vector to be projected.
            - "k": float, radius of the L1 ball.

        Returns
        -------
        dict
            {"solution": projected vector as a list of floats}
        """
        v = np.asarray(problem.get("v", []), dtype=np.float64)
        k = float(problem.get("k", 0.0))

        # Edge cases
        if k <= 0.0:
            w = np.zeros_like(v)
        else:
            abs_v = np.abs(v)
            if np.sum(abs_v) <= k:
                w = v.copy()
            else:
                # Sort absolute values in descending order
                u = np.sort(abs_v)[::-1]
                # Cumulative sum
                cssv = np.cumsum(u)
                # Compute thresholds
                t = (cssv - k) / (np.arange(1, len(u) + 1))
                # Find last index where u > t
                idx = np.where(u > t)[0]
                if idx.size == 0:
                    theta = 0.0
                else:
                    rho = idx[-1]
                    theta = (cssv[rho] - k) / (rho + 1)
                # Soft‑thresholding
                w = np.sign(v) * np.maximum(abs_v - theta, 0.0)

        return {"solution": w.tolist()}
        return {"solution": w.tolist()}