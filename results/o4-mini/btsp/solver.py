import numpy as np
from numba import njit, int32

@njit(cache=True)
def _dp_parent(n, widx):
    """
    Bitmask DP to compute minimal bottleneck index and parent pointers.
    Returns (best_idx, best_u, parent).
    """
    full = (1 << n) - 1
    INF = 1_000_000_000
    # dp[mask, u] = minimal possible max-index of any edge on a path that visits 'mask' ending at u
    dp = np.empty((full + 1, n), dtype=int32)
    parent = np.empty((full + 1, n), dtype=int32)
    # initialize
    for mask in range(full + 1):
        for u in range(n):
            dp[mask, u] = INF
            parent[mask, u] = -1
    dp[1, 0] = 0
    # build DP
    for mask in range(1, full + 1):
        # must include start city 0
        if (mask & 1) == 0:
            continue
        for u in range(n):
            if ((mask >> u) & 1) == 0:
                continue
            val = dp[mask, u]
            if val == INF:
                continue
            for v in range(n):
                if ((mask >> v) & 1) != 0:
                    continue
                idx = widx[u, v]
                # bottleneck along this extension
                new_val = val if val >= idx else idx
                m2 = mask | (1 << v)
                if new_val < dp[m2, v]:
                    dp[m2, v] = new_val
                    parent[m2, v] = u
    # find best end at 0
    best = INF
    best_u = 0
    for u in range(n):
        val = dp[full, u]
        idx0 = widx[u, 0]
        cand = val if val >= idx0 else idx0
        if cand < best:
            best = cand
            best_u = u
    return best, best_u, parent

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n <= 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]
        # collect and index unique weights
        weights = set()
        for i in range(n):
            for j in range(i + 1, n):
                weights.add(problem[i][j])
        wlist = sorted(weights)
        w2i = {w: idx for idx, w in enumerate(wlist)}
        # build index matrix
        widx = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    widx[i, j] = w2i[problem[i][j]]
        # run the bitmask DP
        thr_idx, best_u, parent = _dp_parent(n, widx)
        # reconstruct path
        full = (1 << n) - 1
        mask = full
        u = int(best_u)
        rev = []
        while mask != 1:
            rev.append(u)
            prev = parent[mask, u]
            mask ^= (1 << u)
            u = int(prev)
        path = [0] + rev[::-1] + [0]
        return path