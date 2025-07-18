#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np

def min_bottleneck_idx(np.uint16_t[:, :] widx):
    cdef int n = widx.shape[0]
    cdef int full = (1 << n) - 1
    cdef unsigned short INF = 65535
    cdef np.uint16_t[:, :] dp = np.empty((full + 1, n), dtype=np.uint16)
    dp.fill(INF)
    dp[1, 0] = 0
    cdef int mask, u, v, m2
    cdef unsigned short val, idx, new, best, idx0, cand
    for mask in range(1, full + 1):
        if (mask & 1) == 0:
            continue
        for u in range(n):
            if not ((mask >> u) & 1):
                continue
            val = dp[mask, u]
            if val == INF:
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                idx = widx[u, v]
                m2 = mask | (1 << v)
                new = val if val >= idx else idx
                if new < dp[m2, v]:
                    dp[m2, v] = new
    best = INF
    for u in range(n):
        val = dp[full, u]
        idx0 = widx[u, 0]
        cand = val if val >= idx0 else idx0
        if cand < best:
            best = cand
    return best