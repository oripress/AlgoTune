# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

cpdef list solve_vrp(double[:, :] D, int[:] nodes_arr, int K, int depot):
    cdef int m = nodes_arr.shape[0]
    cdef int M = 1 << m
    cdef double INF = 1e18
    # DP for TSP on subsets
    cdef double[:, :] DP = np.full((M, m), INF, dtype=np.float64)
    cdef int[:, :] PREV = np.full((M, m), -1, dtype=np.int32)
    cdef double[:] cost_s = np.zeros(M, dtype=np.float64)
    cdef int[:] end_node = np.zeros(M, dtype=np.int32)
    cdef int i, j, mask, prev_mask, bit
    cdef double best, cost
    # Initialize single visits
    for i in range(m):
        DP[1<<i, i] = D[depot, nodes_arr[i]]
    # Build DP
    for mask in range(1, M):
        if mask & (mask - 1) == 0:
            continue
        for i in range(m):
            bit = 1 << i
            if not (mask & bit):
                continue
            prev_mask = mask ^ bit
            best = INF
            for j in range(m):
                if (prev_mask >> j) & 1:
                    cost = DP[prev_mask, j] + D[nodes_arr[j], nodes_arr[i]]
                    if cost < best:
                        best = cost
                        PREV[mask, i] = j
            DP[mask, i] = best
    # Close tours back to depot
    cost_s[0] = 0.0; end_node[0] = -1
    for mask in range(1, M):
        best = INF
        for i in range(m):
            if (mask >> i) & 1:
                cost = DP[mask, i] + D[nodes_arr[i], depot]
                if cost < best:
                    best = cost
                    end_node[mask] = i
        cost_s[mask] = best
    # Partition into up to K tours
    cdef int K_eff = K if K <= m else m
    cdef double[:, :] DP2 = np.full((M, K_eff+1), INF, dtype=np.float64)
    cdef int[:, :] choice = np.zeros((M, K_eff+1), dtype=np.int32)
    DP2[0, 0] = 0.0
    for k in range(1, K_eff+1):
        DP2[0, k] = 0.0
    # DP2 for partitioning
    cdef int sub, best_sub, mask2, kk, prev_i
    cdef double val
    for k in range(1, K_eff+1):
        for mask in range(1, M):
            best = INF
            best_sub = 0
            sub = mask
            while sub:
                val = DP2[mask ^ sub, k-1] + cost_s[sub]
                if val < best:
                    best = val
                    best_sub = sub
                sub = (sub - 1) & mask
            DP2[mask, k] = best
            choice[mask, k] = best_sub
    # Reconstruct subsets
    mask2 = M - 1
    kk = K_eff
    cdef list subs = []
    while kk > 0:
        sub = choice[mask2, kk]
        subs.append(sub)
        mask2 ^= sub
        kk -= 1
    # Build routes
    cdef list routes = []
    cdef int ii, temp
    for sub in subs:
        if sub == 0:
            routes.append([depot, depot])
        else:
            path = []
            ii = end_node[sub]
            temp = sub
            while ii != -1:
                path.append(nodes_arr[ii])
                prev_i = PREV[temp, ii]
                temp ^= 1 << ii
                ii = prev_i
            path.reverse()
            routes.append([depot] + path + [depot])
    # Fill empty for unused vehicles
    for _ in range(K - K_eff):
        routes.append([depot, depot])
    return routes