import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] cython_outer(double[::1] vec1, double[::1] vec2):
    cdef int n = vec1.shape[0]
    cdef int m = vec2.shape[0]
    cdef np.ndarray[double, ndim=2] result = np.empty((n, m), dtype=np.double)
    cdef int i, j
    
    for i in range(n):
        for j in range(m):
            result[i, j] = vec1[i] * vec2[j]
    return result