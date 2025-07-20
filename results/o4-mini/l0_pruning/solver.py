import numpy as np
import numba
from numba import njit

@njit(cache=True)
def _l0_prune_nb(v, k):
    n = v.shape[0]
    # trivial cases
    if k <= 0:
        return np.zeros(n)
    if k >= n:
        return v.copy()
    # initialize index array
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i
    left = 0
    right = n - 1
    target = n - k
    # quickselect based on |v|
    while left < right:
        pivot_idx = (left + right) // 2
        # compute pivot magnitude
        pv = v[idx[pivot_idx]]
        pivot = -pv if pv < 0.0 else pv
        # move pivot to end
        tmp_i = idx[pivot_idx]
        idx[pivot_idx] = idx[right]
        idx[right] = tmp_i
        store_idx = left
        for j in range(left, right):
            valj = v[idx[j]]
            absj = -valj if valj < 0.0 else valj
            if absj < pivot:
                tmp_j = idx[j]
                idx[j] = idx[store_idx]
                idx[store_idx] = tmp_j
                store_idx += 1
        # place pivot at its final location
        tmp_s = idx[store_idx]
        idx[store_idx] = idx[right]
        idx[right] = tmp_s
        if store_idx == target:
            break
        elif store_idx < target:
            left = store_idx + 1
        else:
            right = store_idx - 1
    # build output vector
    w = np.zeros(n)
    for j in range(target, n):
        w[idx[j]] = v[idx[j]]
    return w

class Solver:
    def __init__(self):
        # trigger Numba compilation
        _l0_prune_nb(np.zeros(1), 0)

    def solve(self, problem, **kwargs):
        v = np.asarray(problem.get("v", []), dtype=np.float64)
        k = int(problem.get("k", 0))
        w = _l0_prune_nb(v, k)
        return {"solution": w}