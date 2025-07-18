import numpy as np
from typing import Any
from numba import njit

@njit('float64[:](float64[:], int64)', cache=True, fastmath=True, nogil=True)
def _solve(v, k):
    n = len(v)
    if k == 0:
        return np.zeros(n)
    if k >= n:
        return v.copy()
    
    # Get indices of top-k absolute values
    abs_v = np.abs(v)
    topk_indices = np.argpartition(abs_v, n - k)[n - k:]
    
    # Set solution values for top-k indices
    w = np.zeros(n, dtype=np.float64)
    for i in topk_indices:
        w[i] = v[i]
    return w

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem["v"], dtype=np.float64)
        k = problem["k"]
        w = _solve(v, k)
        return {"solution": w.tolist()}