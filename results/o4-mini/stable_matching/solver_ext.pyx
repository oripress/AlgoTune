# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

ctypedef np.int64_t DTYPE_t

def gs(np.ndarray[DTYPE_t, ndim=2] prop not None,
       np.ndarray[DTYPE_t, ndim=2] recv not None):
    cdef int n = prop.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] recv_rank = np.empty((n, n), dtype=np.int64)
    cdef np.ndarray[DTYPE_t, ndim=1] next_prop = np.zeros(n, dtype=np.int64)
    cdef np.ndarray[DTYPE_t, ndim=1] recv_match = np.full(n, -1, dtype=np.int64)
    cdef np.ndarray[DTYPE_t, ndim=1] prop_match = np.full(n, -1, dtype=np.int64)
    cdef np.ndarray[DTYPE_t, ndim=1] free_stack = np.empty(n, dtype=np.int64)
    cdef int i, j, p, r, cur, free_count

    # build receiver ranking table
    for r in range(n):
        for j in range(n):
            recv_rank[r, recv[r, j]] = j

    # initialize free list
    for i in range(n):
        free_stack[i] = i
    free_count = n

    # Gale-Shapley main loop
    while free_count > 0:
        free_count -= 1
        p = free_stack[free_count]
        r = prop[p, next_prop[p]]
        next_prop[p] += 1
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
            prop_match[p] = r
        else:
            if recv_rank[r, p] < recv_rank[r, cur]:
                recv_match[r] = p
                prop_match[p] = r
                prop_match[cur] = -1
                free_stack[free_count] = cur
                free_count += 1
            else:
                free_stack[free_count] = p
                free_count += 1

    # build Python list
    cdef list result = [0] * n
    for i in range(n):
        result[i] = prop_match[i]
    return result