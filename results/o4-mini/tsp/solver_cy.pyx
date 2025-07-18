# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp

cpdef list solve_hk(cnp.ndarray[cnp.int64_t, ndim=2] problem not None):
    """
    Cython-accelerated Held-Karp TSP solver for small n.
    Input: problem as 2D numpy int64 array.
    Output: Python list tour of length n+1.
    """
    cdef int n = problem.shape[0]
    if n <= 1:
        # trivial tour
        return [0, 0]
    cdef int K = n - 1
    cdef int M = 1 << K
    # Large INF
    cdef cnp.int64_t INF = (1 << 60)
    # Allocate dp and parent arrays
    cdef cnp.ndarray[cnp.int64_t, ndim=2] dp = np.empty((M, K), dtype=np.int64)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] parent = np.empty((M, K), dtype=np.int32)
    # Memoryviews for fast access
    cdef cnp.int64_t[:, :] dp_view = dp
    cdef cnp.int32_t[:, :] parent_view = parent
    cdef cnp.int64_t[:, :] pr = problem
    cdef int mask, j, k, new_mask, idx, last
    cdef cnp.int64_t cost_j, new_cost, best
    # Initialize dp and parent
    for mask in range(M):
        for j in range(K):
            dp_view[mask, j] = INF
            parent_view[mask, j] = -1
    # Base cases: from city 0 to j+1
    for j in range(K):
        dp_view[1 << j, j] = pr[0, j + 1]
    # DP state transitions
    for mask in range(M):
        for j in range(K):
            cost_j = dp_view[mask, j]
            if cost_j >= INF:
                continue
            for k in range(K):
                if not (mask & (1 << k)):
                    new_mask = mask | (1 << k)
                    new_cost = cost_j + pr[j + 1, k + 1]
                    if new_cost < dp_view[new_mask, k]:
                        dp_view[new_mask, k] = new_cost
                        parent_view[new_mask, k] = j
    # Find best ending
    full = M - 1
    best = INF
    last = 0
    for j in range(K):
        cost_j = dp_view[full, j] + pr[j + 1, 0]
        if cost_j < best:
            best = cost_j
            last = j
    # Reconstruct tour
    cdef list tour = [0] * (n + 1)
    tour[n] = 0
    mask = full
    j = last
    for idx in range(K - 1, -1, -1):
        tour[idx + 1] = j + 1
        mask_prev = parent_view[mask, j]
        mask ^= (1 << j)
        j = mask_prev
    tour[0] = 0
    return tour