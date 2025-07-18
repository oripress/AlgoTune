# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

cpdef np.ndarray cd_lasso(np.ndarray[double, ndim=2] G,
                          np.ndarray[double, ndim=1] c,
                          double alpha, int max_iter, double tol):
    cdef int d = G.shape[0]
    cdef np.ndarray[double, ndim=1] w_nd = np.zeros(d, dtype=np.double)
    cdef double[:] w = w_nd
    cdef double max_delta, corr, w_new, delta
    cdef int i, j, it
    for it in range(max_iter):
        max_delta = 0.0
        for i in range(d):
            corr = c[i]
            for j in range(d):
                corr -= G[i, j] * w[j]
            corr += G[i, i] * w[i]
            if corr > alpha:
                w_new = (corr - alpha) / G[i, i]
            elif corr < -alpha:
                w_new = (corr + alpha) / G[i, i]
            else:
                w_new = 0.0
            delta = w_new - w[i]
            if delta != 0.0:
                w[i] = w_new
                if delta > max_delta:
                    max_delta = delta
                elif -delta > max_delta:
                    max_delta = -delta
        if max_delta < tol:
            break
    return w_nd