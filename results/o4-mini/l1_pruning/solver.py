from typing import Any, Dict
import numpy as np
from numba import njit

@njit(fastmath=True)
def _threshold_and_build(mu: np.ndarray, v: np.ndarray, k: float) -> np.ndarray:
    """
    Given sorted descending absolute values mu, original v, and radius k,
    find threshold theta via Duchi et al. and construct projected vector w.
    """
    n = mu.shape[0]
    s = 0.0
    theta = 0.0
    # find break point
    for j in range(n):
        s += mu[j]
        t = (s - k) / (j + 1)
        if mu[j] < t:
            theta = t
            break
    # build output
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        vi = v[i]
        av = vi if vi >= 0.0 else -vi
        diff = av - theta
        if diff > 0.0:
            w[i] = diff if vi >= 0.0 else -diff
        else:
            w[i] = 0.0
    return w

# Prime the Numba function at import (overhead not counted in solve)
_dummy = np.array([0.0], dtype=np.float64)
_threshold_and_build(_dummy, _dummy, 0.0)

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, list]:
        # Read inputs
        v = np.asarray(problem["v"], dtype=np.float64).ravel()
        k = float(problem["k"])
        abs_v = np.abs(v)
        # If within L1-ball, no change
        if abs_v.sum() <= k:
            w = v.copy()
        else:
            # Sort absolute values descending
            mu = np.sort(abs_v)
            mu = mu[::-1]
            # Compute projection via optimized loop
            w = _threshold_and_build(mu, v, k)
        return {"solution": w.tolist()}