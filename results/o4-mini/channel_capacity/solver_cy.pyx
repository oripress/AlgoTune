# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
cdef double INF = 1e20

cpdef blahut_arimoto(np.ndarray[np.float64_t, ndim=2] P not None,
                     np.ndarray[np.float64_t, ndim=1] c not None,
                     int max_iter, double tol):
    cdef int m = P.shape[0]
    cdef int n = P.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] x = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] x_new = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(m, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] logy = np.empty(m, dtype=np.float64)
    cdef double C_prev = -INF
    cdef double C_curr = 0.0
    cdef double S
    cdef int it, i, j
    # initialize x uniformly
    for j in range(n):
        x[j] = 1.0 / n
    for it in range(max_iter):
        # compute y = P @ x
        for i in range(m):
            tmp = 0.0
            for j in range(n):
                tmp += P[i, j] * x[j]
            y[i] = tmp
        # compute log2(y)
        for i in range(m):
            if y[i] > 0.0:
                logy[i] = np.log2(y[i])
            else:
                logy[i] = 0.0
        # compute C_curr and update x_new
        C_curr = 0.0
        for j in range(n):
            tmp = 0.0
            for i in range(m):
                tmp += P[i, j] * logy[i]
            d = c[j] - tmp
            x_new[j] = x[j] * (2.0 ** d)
            C_curr += x[j] * d
        # normalize x_new
        S = 0.0
        for j in range(n):
            S += x_new[j]
        if S <= 0.0 or not np.isfinite(S):
            for j in range(n):
                x_new[j] = 1.0 / n
        else:
            for j in range(n):
                x_new[j] /= S
        # check convergence
        if abs(C_curr - C_prev) <= tol:
            for j in range(n):
                x[j] = x_new[j]
            break
        # prepare next iteration
        for j in range(n):
            x[j] = x_new[j]
        C_prev = C_curr
    return x, C_curr