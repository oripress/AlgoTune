# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
cdef double INF = 1e18

def hungarian(np.ndarray[np.float64_t, ndim=2] cost_mat):
    """
    Hungarian algorithm for rectangular cost matrix with n rows <= m columns.
    cost_mat: 2D numpy array of shape (n, m) float64.
    Returns assignments array of length n, where result[i] = assigned column index.
    """
    cdef int n = cost_mat.shape[0]
    cdef int m = cost_mat.shape[1]
    if n > m:
        raise ValueError("hungarian: n > m")
    cdef np.ndarray[np.int64_t, ndim=1] assignment = np.empty(n, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim=1] u = np.zeros(n+1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros(m+1, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] p = np.zeros(m+1, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] way = np.zeros(m+1, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim=1] minv = np.empty(m+1, dtype=np.float64)
    cdef np.ndarray[np.bool_t, ndim=1] used = np.empty(m+1, dtype=np.bool_)
    # memoryviews
    cdef double[:, :] cost = cost_mat
    cdef double[:] u_mv = u
    cdef double[:] v_mv = v
    cdef long[:] p_mv = p
    cdef long[:] way_mv = way
    cdef double[:] minv_mv = minv
    cdef bint[:] used_mv = used
    cdef int i, j, j0, j1, i0
    cdef double delta, cur
    for i in range(1, n+1):
        p_mv[0] = i
        used_mv[0] = True
        minv_mv[0] = 0.0
        for j in range(1, m+1):
            used_mv[j] = False
            minv_mv[j] = INF
        j0 = 0
        while True:
            used_mv[j0] = True
            i0 = p_mv[j0]
            delta = INF
            j1 = 0
            for j in range(1, m+1):
                if not used_mv[j]:
                    cur = cost[i0-1, j-1] - u_mv[i0] - v_mv[j]
                    if cur < minv_mv[j]:
                        minv_mv[j] = cur
                        way_mv[j] = j0
                    if minv_mv[j] < delta:
                        delta = minv_mv[j]
                        j1 = j
            for j in range(0, m+1):
                if used_mv[j]:
                    u_mv[p_mv[j]] += delta
                    v_mv[j] -= delta
                else:
                    minv_mv[j] -= delta
            j0 = j1
            if p_mv[j0] == 0:
                break
        while True:
            j1 = way_mv[j0]
            p_mv[j0] = p_mv[j1]
            j0 = j1
            if j0 == 0:
                break
    # build assignment
    for j in range(1, m+1):
        if p_mv[j] > 0:
            assignment[p_mv[j]-1] = j-1
    return assignment