#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def outer_product_cython(double[::1] vec1, double[::1] vec2):
    cdef int n1 = vec1.shape[0]
    cdef int n2 = vec2.shape[0]
    cdef double[:,::1] result = np.empty((n1, n2), dtype=np.float64)
    cdef int i, j
    cdef double v1, v2
    
    for i in range(n1):
        v1 = vec1[i]
        for j in range(n2):
            result[i, j] = v1 * vec2[j]
    
    return np.asarray(result)