# distutils: language=c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math
# distutils: extra_link_args = -O3 -march=native -ffast-math

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=2] cython_outer(double[:] vec1, double[:] vec2):
    cdef int i, j
    cdef int n = vec1.shape[0]
    cdef int m = vec2.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] result = np.empty((n, m), dtype=np.float64)
    
    for i in prange(n, nogil=True):
        for j in range(m):
            result[i, j] = vec1[i] * vec2[j]
    return result