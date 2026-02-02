# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def construct_Q(np.ndarray[np.float64_t, ndim=1] S, int M):
    cdef int k, l
    cdef int size = M + 1
    cdef np.ndarray[np.float64_t, ndim=2] Q = np.empty((size, size), dtype=np.float64)
    cdef double[:, :] Q_view = Q
    cdef double[:] S_view = S
    
    for k in range(size):
        for l in range(size):
            if k >= l:
                Q_view[k, l] = 0.5 * (S_view[k - l] + S_view[k + l])
            else:
                Q_view[k, l] = 0.5 * (S_view[l - k] + S_view[k + l])
                
    return Q